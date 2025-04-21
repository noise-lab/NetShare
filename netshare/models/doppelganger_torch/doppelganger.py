import os
import sys
import torch
import datetime
import numpy as np
from tqdm import tqdm
# Assuming .network defines the necessary modules
from .network import DoppelGANgerGenerator, Discriminator, AttrDiscriminator
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# Conditional import for Opacus if available
try:
    from opacus.optimizers import DPOptimizer
    from opacus.accountants import RDPAccountant
    from opacus import GradSampleModule
    # Assuming .privacy_util exists relative to this file's location
    from .privacy_util import compute_dp_sgd_privacy
except ImportError:
    print("Opacus or .privacy_util not found or import failed. DP features will not be available.")
    pass
except Exception as e:
    print(f"An unexpected error occurred during Opacus import: {e}")
    pass


class DoppelGANger(object):
    def __init__(
        self,
        # General training related parameters
        checkpoint_dir,
        sample_dir,
        time_path,
        batch_size,
        real_attribute_mask,
        max_sequence_len,
        sample_len,
        data_feature_outputs,
        data_attribute_outputs,
        vis_freq,
        vis_num_sample,
        d_rounds,
        g_rounds,
        d_gp_coe,
        num_packing,
        use_attr_discriminator,
        attr_d_gp_coe,
        g_attr_d_coe,
        epoch_checkpoint_freq,
        attribute_latent_dim,
        feature_latent_dim,
        g_lr,
        g_beta1,
        d_lr,
        d_beta1,
        attr_d_lr,
        attr_d_beta1,
        adam_eps,
        adam_amsgrad,
        # DoppelGANgerGenerator related hyper-parameters
        generator_attribute_num_units,
        generator_attribute_num_layers,
        generator_feature_num_units,
        generator_feature_num_layers,
        use_adaptive_rolling,
        # Discriminator related hyper-parameters
        discriminator_num_layers,
        discriminator_num_units,
        # Attr discriminator related hyper-parameters
        # Please ignore these params if use_attr_discriminator = False
        attr_discriminator_num_layers,
        attr_discriminator_num_units,
        # Pretrain-related
        restore=False,
        pretrain_dir=None
    ):

        # --- Store all configuration parameters ---
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.time_path = time_path
        self.batch_size = batch_size
        self.real_attribute_mask = real_attribute_mask
        self.max_sequence_len = max_sequence_len
        self.sample_len = sample_len
        self.data_feature_outputs = data_feature_outputs
        self.data_attribute_outputs = data_attribute_outputs
        self.vis_freq = vis_freq
        self.vis_num_sample = vis_num_sample
        self.num_packing = num_packing
        self.use_attr_discriminator = use_attr_discriminator
        self.d_rounds = d_rounds
        self.g_rounds = g_rounds
        self.d_gp_coe = d_gp_coe
        self.attr_d_gp_coe = attr_d_gp_coe
        self.g_attr_d_coe = g_attr_d_coe
        self.epoch_checkpoint_freq = epoch_checkpoint_freq
        self.attribute_latent_dim = attribute_latent_dim
        self.feature_latent_dim = feature_latent_dim
        self.g_lr = g_lr
        self.g_beta1 = g_beta1
        self.d_lr = d_lr
        self.d_beta1 = d_beta1
        self.attr_d_lr = attr_d_lr
        self.attr_d_beta1 = attr_d_beta1
        self.adam_eps = adam_eps
        self.adam_amsgrad = adam_amsgrad
        self.generator_attribute_num_units = generator_attribute_num_units
        self.generator_attribute_num_layers = generator_attribute_num_layers
        self.generator_feature_num_units = generator_feature_num_units
        self.generator_feature_num_layers = generator_feature_num_layers
        self.use_adaptive_rolling = use_adaptive_rolling
        self.discriminator_num_layers = discriminator_num_layers
        self.discriminator_num_units = discriminator_num_units
        self.attr_discriminator_num_layers = attr_discriminator_num_layers
        self.attr_discriminator_num_units = attr_discriminator_num_units
        self.restore = restore
        self.pretrain_dir = pretrain_dir

        self.EPS = 1e-8 # Epsilon for numerical stability
        self.MODEL_NAME = "model" # Default model name prefix

        # Set device to CUDA if available, otherwise CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Validate sequence length configuration
        if self.max_sequence_len % self.sample_len != 0:
            raise ValueError("max_sequence_len must be a multiple of sample_len")

        # Calculate number of samples in time dimension
        self.sample_time = self.max_sequence_len // self.sample_len

        self.is_build = False # Flag to check if model components are built

        # Calculate total feature and attribute dimensions
        self.feature_dim = np.sum([t.dim for t in self.data_feature_outputs])
        self.attribute_dim = np.sum(
            [t.dim for t in self.data_attribute_outputs])

        # --- FIX: Initialize gen_flag_dims here ---
        self.gen_flag_dims = None
        current_dim = 0
        for output in self.data_feature_outputs:
            if output.is_gen_flag:
                if output.dim != 2:
                    # Raise error immediately if config is wrong
                    raise ValueError(f"Generation flag feature '{output.name}' must have dim=2, but got {output.dim}")
                self.gen_flag_dims = [current_dim, current_dim + output.dim]
                # print(f"Found generation flag feature '{output.name}' at dimensions: {self.gen_flag_dims}")
                break # Stop after finding the first flag
            current_dim += output.dim

        # Check if gen_flag_dims was found
        if self.gen_flag_dims is None:
            raise ValueError("Generation flag feature (is_gen_flag=True) not found in data_feature_outputs configuration.")
        # --- End FIX ---

        # Build the model components
        self._build()

        # Initialize writer (will be properly set in train)
        self.writer = None


    def check_data(self):
        """
        Validates the structure and dimensions of the input data against model configuration.
        This is typically called during training.
        """
        # gen_flag_dims is now initialized in __init__, so we just check data shapes here.

        # Check if feature dimensions match configuration
        if not hasattr(self, 'data_feature') or self.data_feature is None:
             print("Warning: Skipping feature dimension check in check_data() as self.data_feature is not set.")
        elif self.data_feature.shape[2] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch in training data: expected {self.feature_dim}, got {self.data_feature.shape[2]}"
            )

        # Check generation flag shape (should be N x T)
        if not hasattr(self, 'data_gen_flag') or self.data_gen_flag is None:
             print("Warning: Skipping data_gen_flag shape check in check_data() as self.data_gen_flag is not set.")
        elif len(self.data_gen_flag.shape) != 2:
            raise ValueError("Training data_gen_flag should have 2 dimensions (samples, sequence_length)")

        print("Data checks passed (or skipped where data not present).")


    def train(self, epochs, data_feature, data_attribute, data_gen_flag):
        """
        Main training loop for the DoppelGANger model.

        Args:
            epochs (int): Number of training epochs.
            data_feature (np.ndarray): Feature data (samples, seq_len, feature_dim).
            data_attribute (np.ndarray): Attribute data (samples, attribute_dim).
            data_gen_flag (np.ndarray): Generation flags (samples, seq_len).
        """
        self.epochs = epochs
        # Store references to the training data
        self.data_feature = data_feature
        self.data_attribute = data_attribute
        self.data_gen_flag = data_gen_flag

        # Setup TensorBoard writer in a timestamped subdirectory
        log_dir = os.path.join(self.checkpoint_dir, "runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True) # Ensure log directory exists
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")

        # Perform data validation checks using the stored data
        self.check_data()

        # Create PyTorch dataset and dataloader
        dataset = TensorDataset(
            torch.from_numpy(data_attribute).float(),
            torch.from_numpy(data_feature).float()
        )

        # Start the internal training process
        self._train(dataset)

        # Close the TensorBoard writer after training
        if self.writer:
            self.writer.close()
            print("TensorBoard writer closed.")


    def generate(
        self,
        num_samples,
        given_attribute=None,
        given_attribute_discrete=None,
        return_gen_flag_feature=False,
        # --- Optional: Pass writer for logging during generation ---
        writer=None
        # --- End Optional ---
    ):
        """
        Generates synthetic data samples using the trained generator.

        Args:
            num_samples (int): Number of samples to generate.
            given_attribute (np.ndarray, optional): Continuous attributes to condition on. Defaults to None.
            given_attribute_discrete (np.ndarray, optional): Discrete attributes to condition on. Defaults to None.
            return_gen_flag_feature (bool, optional): Whether to include the gen flag feature in the output. Defaults to False.
            writer (SummaryWriter, optional): TensorBoard writer for logging generation losses. Defaults to None (uses self.writer if available).

        Returns:
            tuple: (feature, attribute, attribute_discrete, gen_flag, lengths)
                   - feature (np.ndarray): Generated features.
                   - attribute (np.ndarray): Generated continuous attributes.
                   - attribute_discrete (np.ndarray): Generated discrete attributes.
                   - gen_flag (np.ndarray): Generated generation flags (binary).
                   - lengths (np.ndarray): Calculated sequence lengths based on gen_flag.
        """
        if not self.is_build:
            raise RuntimeError("Model has not been built or trained yet. Call train() or load() first.")
        # Ensure gen_flag_dims was initialized correctly
        if self.gen_flag_dims is None:
             raise RuntimeError("gen_flag_dims was not initialized correctly. Check data_feature_outputs configuration.")


        # --- Use existing writer or the one passed as argument ---
        log_dir = os.path.join(self.checkpoint_dir, "generation", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True) # Ensure log directory exists
        log_writer = SummaryWriter(log_dir=log_dir)
        # --- End writer handling ---

        # Calculate number of batches needed
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size # Ceiling division
        print(f"Generating {num_samples} samples in {num_batches} batches (batch size: {self.batch_size})...")

        # Generate noise inputs for the generator
        real_attribute_noise = self._gen_attribute_input_noise(num_samples).to(self.device)
        addi_attribute_noise = self._gen_attribute_input_noise(num_samples).to(self.device)
        feature_input_noise = self._gen_feature_input_noise(num_samples, self.sample_time).to(self.device)

        # Generate initial hidden states for the feature generator RNN
        h0 = torch.randn(self.generator.feature_num_layers, num_samples, self.generator.feature_num_units, device=self.device)
        c0 = torch.randn(self.generator.feature_num_layers, num_samples, self.generator.feature_num_units, device=self.device)

        generated_data_list = []
        # Generate data batch by batch
        for n_batch in tqdm(range(num_batches), desc="Generating Batches"):
            start_idx = n_batch * self.batch_size
            end_idx = min((n_batch + 1) * self.batch_size, num_samples)
            # current_batch_size = end_idx - start_idx # Not needed if slicing handles last batch correctly

            # Slice noise and hidden states for the current batch
            batch_real_attr_noise = real_attribute_noise[start_idx:end_idx]
            batch_addi_attr_noise = addi_attribute_noise[start_idx:end_idx]
            batch_feat_noise = feature_input_noise[start_idx:end_idx]
            batch_h0 = h0[:, start_idx:end_idx, :]
            batch_c0 = c0[:, start_idx:end_idx, :]

            # Slice conditional attributes if provided
            batch_given_attribute = None
            batch_given_attribute_discrete = None
            if given_attribute is not None:
                # Ensure it's a NumPy array before slicing
                if isinstance(given_attribute, torch.Tensor):
                     given_attribute = given_attribute.cpu().numpy()
                batch_given_attribute = given_attribute[start_idx:end_idx]
            if given_attribute_discrete is not None:
                 if isinstance(given_attribute_discrete, torch.Tensor):
                      given_attribute_discrete = given_attribute_discrete.cpu().numpy()
                 batch_given_attribute_discrete = given_attribute_discrete[start_idx:end_idx]

            # Call the internal _generate method for the current batch
            generated_batch_data = self._generate(
                real_attribute_noise=batch_real_attr_noise,
                addi_attribute_noise=batch_addi_attr_noise,
                feature_input_noise=batch_feat_noise,
                h0=batch_h0,
                c0=batch_c0,
                # --- Pass logging arguments ---
                iteration=n_batch,  # Use batch index as iteration for generation logging
                writer=log_writer,
                # --- End logging arguments ---
                given_attribute=batch_given_attribute, # Pass as numpy or tensor, _generate handles conversion
                given_attribute_discrete=batch_given_attribute_discrete
            )
            generated_data_list.append(generated_batch_data) # Append tuple (attribute, attribute_discrete, feature)

        # Concatenate results from all batches
        if not generated_data_list:
             print("Warning: No data generated.")
             return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]) # Return empty arrays

        # Unzip the list of tuples and concatenate each part
        attribute_list, attribute_discrete_list, feature_list = zip(*generated_data_list)
        attribute = np.concatenate(attribute_list, axis=0)
        attribute_discrete = np.concatenate(attribute_discrete_list, axis=0)
        feature = np.concatenate(feature_list, axis=0)


        # Post-process generated features to determine sequence lengths based on gen_flag
        # Extract the generation flag feature using the pre-calculated indices
        gen_flag_feature = feature[:, :, self.gen_flag_dims[0]:self.gen_flag_dims[1]]

        # Convert softmax output (if applicable) to binary flag
        # Assuming the flag indicating "generate" is the first part of the 2-dim flag feature
        # If the second part indicates "stop", use argmax or compare probabilities.
        # Using round on the first dim is a common heuristic if it represents P(generate).
        gen_flag = np.round(gen_flag_feature[:, :, 0]) # Example: use first dim, round probability

        # Ensure sequences terminate correctly based on the first '0' flag
        gen_flag_helper = np.concatenate(
            [gen_flag, np.zeros((gen_flag.shape[0], 1))], axis=1
        )
        min_indicator = np.argmin(gen_flag_helper, axis=1)
        for row, min_idx in enumerate(min_indicator):
            gen_flag[row, min_idx:] = 0.0
        lengths = np.sum(gen_flag, axis=1).astype(int)

        # Optionally remove the generation flag feature from the output
        if not return_gen_flag_feature:
            feature_mask = np.ones(feature.shape[2], dtype=bool)
            feature_mask[self.gen_flag_dims[0]:self.gen_flag_dims[1]] = False
            feature = feature[:, :, feature_mask]

        print("Generation complete.")
        return feature, attribute, attribute_discrete, gen_flag, lengths


    def save(self, file_path, only_generator=False, include_optimizer=False):
        """
        Saves the model state to a file.

        Args:
            file_path (str): Path to save the model checkpoint.
            only_generator (bool, optional): If True, save only the generator state. Defaults to False.
            include_optimizer (bool, optional): If True, include optimizer states. Defaults to False.
        """
        print(f"Saving model state to {file_path}...")
        if only_generator:
            state = {
                "generator_state_dict": self.generator.state_dict(),
            }
            print("Saving only generator state.")
        else:
            state = {
                "generator_state_dict": self.generator.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
            }
            if self.use_attr_discriminator and self.attr_discriminator:
                state[
                    "attr_discriminator_state_dict"
                ] = self.attr_discriminator.state_dict()
            print("Saving generator, discriminator, and attribute discriminator (if applicable) states.")

            if include_optimizer:
                 # Check if optimizers exist before saving
                 if hasattr(self, 'opt_generator') and self.opt_generator:
                     state["generator_optimizer_state_dict"] = self.opt_generator.state_dict()
                 else:
                      print("Warning: Generator optimizer not found, cannot save its state.")

                 if hasattr(self, 'opt_discriminator') and self.opt_discriminator:
                     state["discriminator_optimizer_state_dict"] = self.opt_discriminator.state_dict()
                 else:
                      print("Warning: Discriminator optimizer not found, cannot save its state.")


                 if self.use_attr_discriminator and hasattr(self, 'opt_attr_discriminator') and self.opt_attr_discriminator:
                     state["attr_discriminator_optimizer_state_dict"] = self.opt_attr_discriminator.state_dict()
                 elif self.use_attr_discriminator:
                      print("Warning: Attribute Discriminator optimizer not found, cannot save its state.")

                 print("Including optimizer states (if found).")
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(state, file_path)
        print("Model saved successfully.")


    def load(self, model_path):
        """
        Loads the model state from a file.

        Args:
            model_path (str): Path to the model checkpoint file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint file not found: {model_path}")

        print(f"Loading model state from {model_path}...")
        # Load state dict onto the correct device immediately
        state = torch.load(model_path, map_location=self.device)

        # Load generator state
        if "generator_state_dict" in state:
            self.generator.load_state_dict(state["generator_state_dict"])
            print("Generator state loaded.")
        else:
             print("Warning: Generator state not found in checkpoint.")

        # Load discriminator state if present
        if "discriminator_state_dict" in state:
            self.discriminator.load_state_dict(state["discriminator_state_dict"])
            print("Discriminator state loaded.")
        else:
            print("Warning: Discriminator state not found in checkpoint.")


        # Load attribute discriminator state if applicable and present
        if self.use_attr_discriminator:
            if self.attr_discriminator is None:
                 print("Warning: use_attr_discriminator is True, but self.attr_discriminator model is None. Rebuilding...")
                 # Potentially rebuild the attr_discriminator here if needed based on config
                 # self._build() # Or just the relevant part
            elif "attr_discriminator_state_dict" in state:
                self.attr_discriminator.load_state_dict(
                    state["attr_discriminator_state_dict"]
                )
                print("Attribute Discriminator state loaded.")
            else:
                 print("Warning: Attribute Discriminator state not found in checkpoint (but use_attr_discriminator is True).")


        # Load optimizer states if present and optimizers exist
        # Note: Loading optimizer state is usually only needed if resuming training
        if "generator_optimizer_state_dict" in state:
            if hasattr(self, 'opt_generator') and self.opt_generator:
                try:
                    self.opt_generator.load_state_dict(
                        state["generator_optimizer_state_dict"])
                    print("Generator optimizer state loaded.")
                except Exception as e:
                    print(f"Warning: Could not load generator optimizer state: {e}")
            else:
                 print("Warning: Generator optimizer state found in checkpoint, but optimizer object doesn't exist.")

        if "discriminator_optimizer_state_dict" in state:
            if hasattr(self, 'opt_discriminator') and self.opt_discriminator:
                try:
                    self.opt_discriminator.load_state_dict(
                        state["discriminator_optimizer_state_dict"])
                    print("Discriminator optimizer state loaded.")
                except Exception as e:
                    print(f"Warning: Could not load discriminator optimizer state: {e}")

            else:
                 print("Warning: Discriminator optimizer state found in checkpoint, but optimizer object doesn't exist.")


        if self.use_attr_discriminator and "attr_discriminator_optimizer_state_dict" in state:
            if hasattr(self, 'opt_attr_discriminator') and self.opt_attr_discriminator:
                try:
                    self.opt_attr_discriminator.load_state_dict(
                        state["attr_discriminator_optimizer_state_dict"])
                    print("Attribute Discriminator optimizer state loaded.")
                except Exception as e:
                     print(f"Warning: Could not load attribute discriminator optimizer state: {e}")

            else:
                print("Warning: Attribute Discriminator optimizer state found in checkpoint, but optimizer object doesn't exist.")

        print("Model loaded successfully.")
        self.is_build = True # Mark model as ready after loading


    def _build(self):
        """
        Builds the Generator, Discriminator, and optionally Attribute Discriminator networks
        and their corresponding optimizers.
        """
        print("Building model components...")
        # Build Generator
        self.generator = DoppelGANgerGenerator(
            attr_latent_dim=self.attribute_latent_dim,
            feature_latent_dim=self.feature_latent_dim,
            feature_outputs=self.data_feature_outputs,
            attribute_outputs=self.data_attribute_outputs,
            real_attribute_mask=self.real_attribute_mask,
            sample_len=self.sample_len,
            attribute_num_units=self.generator_attribute_num_units,
            attribute_num_layers=self.generator_attribute_num_layers,
            feature_num_units=self.generator_feature_num_units,
            feature_num_layers=self.generator_feature_num_layers,
            batch_size=self.batch_size, # Note: Generator might need batch size info internally
            use_adaptive_rolling=self.use_adaptive_rolling,
            device=self.device
        ).to(self.device)

        # Build Discriminator
        self.discriminator = Discriminator(
            max_sequence_len=self.max_sequence_len, # Might not be needed if input shapes are dynamic
            input_feature_dim=self.feature_dim * self.num_packing,
            input_attribute_dim=self.attribute_dim * self.num_packing,
            num_layers=self.discriminator_num_layers,
            num_units=self.discriminator_num_units,
        ).to(self.device)

        # Build Attribute Discriminator if enabled
        self.attr_discriminator = None # Initialize as None
        if self.use_attr_discriminator:
            self.attr_discriminator = AttrDiscriminator(
                input_attribute_dim=self.attribute_dim * self.num_packing,
                num_layers=self.attr_discriminator_num_layers,
                num_units=self.attr_discriminator_num_units,
            ).to(self.device)

        # Setup Optimizers
        self.opt_generator = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.g_lr,
            betas=(self.g_beta1, 0.999),
            eps=self.adam_eps,
            amsgrad=self.adam_amsgrad,
        )
        self.opt_discriminator = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.d_lr,
            betas=(self.d_beta1, 0.999),
            eps=self.adam_eps,
            amsgrad=self.adam_amsgrad,
        )

        self.opt_attr_discriminator = None # Initialize as None
        if self.use_attr_discriminator and self.attr_discriminator:
            self.opt_attr_discriminator = torch.optim.Adam(
                self.attr_discriminator.parameters(),
                lr=self.attr_d_lr,
                betas=(self.attr_d_beta1, 0.999),
                eps=self.adam_eps,
                amsgrad=self.adam_amsgrad,
            )

        self.is_build = True
        print("Model components built successfully.")


    def _gen_attribute_input_noise(self, batch_size):
        """Generates random noise for the attribute generator input."""
        return torch.randn(int(batch_size), self.attribute_latent_dim) # Ensure batch_size is int


    def _gen_feature_input_noise(self, batch_size, length):
        """Generates random noise for the feature generator input."""
        return torch.randn(int(batch_size), int(length), self.feature_latent_dim) # Ensure ints


    def _calculate_gp_dis(
            self, batch_size, fake_feature, real_feature, fake_attribute,
            real_attribute):
        """
        Calculates the Gradient Penalty for the main Discriminator (WGAN-GP).
        """
        # Sample random interpolation factor alpha
        alpha_dim2 = torch.rand(batch_size, 1, device=self.device) # Shape (batch_size, 1)
        alpha_dim3 = alpha_dim2.unsqueeze(2) # Shape (batch_size, 1, 1) for feature broadcasting

        # Interpolate features and attributes
        interpolates_input_feature = (
            real_feature + alpha_dim3 * (fake_feature - real_feature)
        ).requires_grad_(True)
        interpolates_input_attribute = (
            real_attribute + alpha_dim2 * (fake_attribute - real_attribute)
        ).requires_grad_(True)

        # Get discriminator output for interpolated samples
        mixed_scores = self.discriminator(
            interpolates_input_feature, interpolates_input_attribute
        )

        # Calculate gradients w.r.t. the interpolated inputs
        gradients = torch.autograd.grad(
            inputs=[interpolates_input_feature, interpolates_input_attribute],
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores, device=self.device),
            create_graph=True, # Important for GP calculation within training loop
            retain_graph=True, # Important if graph is needed later
            only_inputs=True, # We only need gradients w.r.t inputs
        )

        # Calculate gradient norms
        gradient_feature_norm_sq = torch.sum(gradients[0]**2, dim=(1, 2))
        gradient_attribute_norm_sq = torch.sum(gradients[1]**2, dim=1)

        # Combined gradient norm
        slopes = torch.sqrt(gradient_feature_norm_sq + gradient_attribute_norm_sq + self.EPS)

        # Calculate Gradient Penalty loss: mean squared difference from 1
        dis_loss_gp = torch.mean((slopes - 1.0) ** 2)

        return dis_loss_gp


    def _calculate_gp_attr_dis(
            self, batch_size, fake_attribute, real_attribute):
        """
        Calculates the Gradient Penalty for the Attribute Discriminator (WGAN-GP).
        """
        if not (self.use_attr_discriminator and self.attr_discriminator):
            return torch.tensor(0.0, device=self.device) # Return zero if not used

        # Sample random interpolation factor alpha
        alpha_dim2 = torch.rand(batch_size, 1, device=self.device) # Shape (batch_size, 1)

        # Interpolate attributes
        interpolates_input_attribute = (
            real_attribute + alpha_dim2 * (fake_attribute - real_attribute)
        ).requires_grad_(True)

        # Get attribute discriminator output for interpolated samples
        mixed_scores = self.attr_discriminator(interpolates_input_attribute)

        # Calculate gradients w.r.t. the interpolated attributes
        gradients = torch.autograd.grad(
            inputs=interpolates_input_attribute,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores, device=self.device),
            create_graph=True, # Important for GP calculation
            retain_graph=True, # Important if graph is needed later
            only_inputs=True,
        )[0] # Get the first (and only) gradient tensor

        # Calculate gradient norms (L2 norm)
        slopes = torch.sqrt(torch.sum(gradients**2, dim=1) + self.EPS)

        # Calculate Gradient Penalty loss
        attr_dis_gp = torch.mean((slopes - 1.0) ** 2)

        return attr_dis_gp


    def _train(self, dataset):
        """Internal training loop logic."""
        # Restore model if specified
        if self.restore:
            if self.pretrain_dir is None:
                raise ValueError("restore=True but pretrain_dir is not specified.")
            if not os.path.exists(self.pretrain_dir):
                raise FileNotFoundError(f"Pretrain directory not found: {self.pretrain_dir}")
            print(f"Restoring pre-trained model from {self.pretrain_dir}...")
            self.load(self.pretrain_dir) # Load model state

        # Ensure models are in training mode
        self.generator.train()
        self.discriminator.train()
        if self.use_attr_discriminator and self.attr_discriminator:
            self.attr_discriminator.train()

        # Create DataLoader
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size * self.num_packing, # Adjust batch size for packing
            shuffle=True,
            num_workers=min(os.cpu_count(), 4), # Use reasonable number of workers
            pin_memory=torch.cuda.is_available(), # Pin memory if using CUDA
            drop_last=True, # Drop last incomplete batch
            # prefetch_factor=2, # Can tune this
            persistent_workers=True if min(os.cpu_count(), 4) > 0 else False, # Keep workers alive if using them
        )

        iteration = 0 # Global iteration counter
        # Initialize loss dictionary with zero tensors on the correct device
        loss_dict = {
            k: torch.tensor(0.0, device=self.device)
            for k in [
                "g_loss_d", "g_loss", "d_loss_fake", "d_loss_real",
                "d_loss_gp", "d_loss"
            ]
        }
        if self.use_attr_discriminator:
            loss_dict.update({
                k: torch.tensor(0.0, device=self.device)
                for k in [
                    "g_loss_attr_d", "attr_d_loss_fake", "attr_d_loss_real",
                    "attr_d_loss_gp", "attr_d_loss"
                ]
            })

        print(f"Starting training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            epoch_start_time = datetime.datetime.now()
            print(f"\nEpoch {epoch+1}/{self.epochs} starting at {epoch_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            # Log epoch start time
            if self.time_path:
                try:
                    os.makedirs(os.path.dirname(self.time_path), exist_ok=True)
                    with open(self.time_path, "a") as f:
                        f.write(f"Epoch {epoch+1} starts: {epoch_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                except IOError as e:
                    print(f"Warning: Could not write to time log file {self.time_path}: {e}")


            batch_iterator = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}", leave=False)
            for batch_idx, (real_attribute_packed, real_feature_packed) in batch_iterator:

                # Move real data to the device
                real_attribute_packed = real_attribute_packed.to(self.device) # (B*P, A_dim)
                real_feature_packed = real_feature_packed.to(self.device)     # (B*P, T, F_dim)

                # Effective batch size (number of packed groups)
                current_batch_size = real_attribute_packed.size(0) // self.num_packing
                if current_batch_size == 0: continue # Skip if batch size is zero after packing

                # --- Train Discriminator(s) ---
                for _ in range(self.d_rounds):
                     # Zero gradients for discriminators
                    self.opt_discriminator.zero_grad()
                    if self.use_attr_discriminator and self.opt_attr_discriminator:
                         self.opt_attr_discriminator.zero_grad()

                    # Generate fake data for the discriminator update
                    with torch.no_grad(): # No need to track generator gradients here
                        real_attribute_noise = self._gen_attribute_input_noise(current_batch_size * self.num_packing).to(self.device)
                        addi_attribute_noise = self._gen_attribute_input_noise(current_batch_size * self.num_packing).to(self.device)
                        feature_input_noise = self._gen_feature_input_noise(current_batch_size * self.num_packing, self.sample_time).to(self.device)
                        h0 = torch.randn(self.generator.feature_num_layers, current_batch_size * self.num_packing, self.generator.feature_num_units, device=self.device)
                        c0 = torch.randn(self.generator.feature_num_layers, current_batch_size * self.num_packing, self.generator.feature_num_units, device=self.device)

                        fake_attribute_packed, _, fake_feature_packed = self.generator(
                            real_attribute_noise, addi_attribute_noise, feature_input_noise, h0, c0
                        ) # Shapes: (B*P, A_dim), (B*P, T, F_dim)

                    # Reshape real and fake data for discriminator input (packing)
                    # Attributes: (B, P * A_dim)
                    packed_real_attribute = real_attribute_packed.view(current_batch_size, -1)
                    packed_fake_attribute = fake_attribute_packed.view(current_batch_size, -1)
                    # Features: (B, P * T, F_dim) - Assuming packing along time dimension
                    packed_real_feature = real_feature_packed.view(current_batch_size, self.num_packing * self.max_sequence_len, self.feature_dim)
                    packed_fake_feature = fake_feature_packed.view(current_batch_size, self.num_packing * self.max_sequence_len, self.feature_dim)


                    # --- Main Discriminator Update ---
                    dis_real = self.discriminator(packed_real_feature, packed_real_attribute)
                    dis_fake = self.discriminator(packed_fake_feature.detach(), packed_fake_attribute.detach())

                    dis_loss_fake = torch.mean(dis_fake)
                    dis_loss_real = -torch.mean(dis_real)
                    dis_loss_gp = self._calculate_gp_dis(current_batch_size, packed_fake_feature, packed_real_feature, packed_fake_attribute, packed_real_attribute)
                    dis_loss = dis_loss_fake + dis_loss_real + self.d_gp_coe * dis_loss_gp

                    dis_loss.backward() # Calculate gradients for D
                    self.opt_discriminator.step() # Update D

                    loss_dict["d_loss_fake"] = dis_loss_fake
                    loss_dict["d_loss_real"] = dis_loss_real
                    loss_dict["d_loss_gp"] = dis_loss_gp
                    loss_dict["d_loss"] = dis_loss

                    # --- Attribute Discriminator Update (if used) ---
                    if self.use_attr_discriminator and self.attr_discriminator and self.opt_attr_discriminator:
                        attr_dis_real = self.attr_discriminator(packed_real_attribute)
                        attr_dis_fake = self.attr_discriminator(packed_fake_attribute.detach())

                        attr_dis_loss_fake = torch.mean(attr_dis_fake)
                        attr_dis_loss_real = -torch.mean(attr_dis_real)
                        attr_dis_loss_gp = self._calculate_gp_attr_dis(current_batch_size, packed_fake_attribute, packed_real_attribute)
                        attr_dis_loss = attr_dis_loss_fake + attr_dis_loss_real + self.attr_d_gp_coe * attr_dis_loss_gp

                        attr_dis_loss.backward() # Calculate gradients for Attr D
                        self.opt_attr_discriminator.step() # Update Attr D

                        loss_dict["attr_d_loss_fake"] = attr_dis_loss_fake
                        loss_dict["attr_d_loss_real"] = attr_dis_loss_real
                        loss_dict["attr_d_loss_gp"] = attr_dis_loss_gp
                        loss_dict["attr_d_loss"] = attr_dis_loss

                # --- Train Generator ---
                for _ in range(self.g_rounds):
                     self.opt_generator.zero_grad() # Zero G gradients

                     # Generate new batch of fake data (track gradients for G)
                     real_attribute_noise = self._gen_attribute_input_noise(current_batch_size * self.num_packing).to(self.device)
                     addi_attribute_noise = self._gen_attribute_input_noise(current_batch_size * self.num_packing).to(self.device)
                     feature_input_noise = self._gen_feature_input_noise(current_batch_size * self.num_packing, self.sample_time).to(self.device)
                     h0 = torch.randn(self.generator.feature_num_layers, current_batch_size * self.num_packing, self.generator.feature_num_units, device=self.device)
                     c0 = torch.randn(self.generator.feature_num_layers, current_batch_size * self.num_packing, self.generator.feature_num_units, device=self.device)

                     fake_attribute_packed, _, fake_feature_packed = self.generator(
                         real_attribute_noise, addi_attribute_noise, feature_input_noise, h0, c0
                     )
                     # Reshape for discriminators
                     packed_fake_attribute = fake_attribute_packed.view(current_batch_size, -1)
                     packed_fake_feature = fake_feature_packed.view(current_batch_size, self.num_packing * self.max_sequence_len, self.feature_dim)

                     # Calculate G loss based on D scores
                     dis_fake = self.discriminator(packed_fake_feature, packed_fake_attribute)
                     gen_loss_dis = -torch.mean(dis_fake) # G wants to maximize D's score for fake

                     # Calculate G loss based on Attr D scores (if used)
                     gen_loss_attr_dis = torch.tensor(0.0, device=self.device)
                     if self.use_attr_discriminator and self.attr_discriminator:
                         attr_dis_fake = self.attr_discriminator(packed_fake_attribute)
                         gen_loss_attr_dis = -torch.mean(attr_dis_fake)
                         loss_dict["g_loss_attr_d"] = gen_loss_attr_dis # Store component

                     # Total Generator loss
                     gen_loss = gen_loss_dis + self.g_attr_d_coe * gen_loss_attr_dis

                     gen_loss.backward() # Calculate gradients for G
                     self.opt_generator.step() # Update G

                     loss_dict["g_loss_d"] = gen_loss_dis
                     loss_dict["g_loss"] = gen_loss


                # --- Logging ---
                if self.writer and iteration % self.vis_freq == 0: # Log every vis_freq iterations
                     self._write_losses(loss_dict, iteration)

                # Update progress bar description (less frequently to reduce overhead)
                if iteration % 10 == 0:
                    log_str = f"Epoch {epoch+1} It {iteration} ["
                    log_str += f"D Loss: {loss_dict['d_loss'].item():.4f} "
                    log_str += f"G Loss: {loss_dict['g_loss'].item():.4f}"
                    if self.use_attr_discriminator:
                         log_str += f" Attr D Loss: {loss_dict['attr_d_loss'].item():.4f}"
                    log_str += "]"
                    batch_iterator.set_description(log_str)


                iteration += 1 # Increment global iteration counter

            # --- End of Epoch ---
            batch_iterator.close() # Close tqdm iterator for the epoch
            epoch_end_time = datetime.datetime.now()
            print(f"Epoch {epoch+1} finished at {epoch_end_time.strftime('%Y-%m-%d %H:%M:%S')}. Duration: {epoch_end_time - epoch_start_time}")

            # Save checkpoint periodically
            if (epoch + 1) % self.epoch_checkpoint_freq == 0:
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"epoch_id-{epoch+1}.pt")
                self.save(ckpt_path, include_optimizer=True) # Include optimizer state

            # Log epoch end time
            if self.time_path:
                 try:
                     with open(self.time_path, "a") as f:
                         f.write(f"Epoch {epoch+1} ends: {epoch_end_time.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                 except IOError as e:
                     print(f"Warning: Could not write to time log file {self.time_path}: {e}")

        print("Training finished.")
        # Save final model (without optimizer state by default)
        final_model_path = os.path.join(self.checkpoint_dir, "final_model.pt")
        self.save(final_model_path, include_optimizer=False)


    def _generate(
        self,
        real_attribute_noise,
        addi_attribute_noise,
        feature_input_noise,
        h0,
        c0,
        # --- Added arguments for logging ---
        iteration, # Iteration number (e.g., generation batch index)
        writer,    # Tensorboard writer instance (can be None)
        # --- End added arguments ---
        given_attribute=None,
        given_attribute_discrete=None,
    ):
        """
        Internal method to generate a single batch of data and optionally log losses.
        """
        # Ensure models are in evaluation mode
        self.generator.eval()
        self.discriminator.eval()
        if self.use_attr_discriminator and self.attr_discriminator:
            self.attr_discriminator.eval()

        # Generate data using the generator
        with torch.no_grad(): # Disable gradient calculation for generation
            # Ensure conditional inputs are tensors on the correct device if provided
            if given_attribute is not None and not isinstance(given_attribute, torch.Tensor):
                given_attribute = torch.from_numpy(given_attribute).float().to(self.device)
            if given_attribute_discrete is not None and not isinstance(given_attribute_discrete, torch.Tensor):
                 given_attribute_discrete = torch.from_numpy(given_attribute_discrete).float().to(self.device)

            # Generate features and attributes
            attribute, attribute_discrete, feature = self.generator(
                real_attribute_noise=real_attribute_noise, # Assumed on device from caller
                addi_attribute_noise=addi_attribute_noise, # Assumed on device
                feature_input_noise=feature_input_noise,   # Assumed on device
                h0=h0.contiguous(), # Ensure contiguous memory
                c0=c0.contiguous(),
                given_attribute=given_attribute, # Pass tensors or None
                given_attribute_discrete=given_attribute_discrete
            )

        # --- Added Loss Calculation and Writing for Generation ---
        if writer is not None: # Only calculate and log if a writer is provided
            loss_dict_gen = {}
            with torch.no_grad(): # Still no gradients needed for scoring
                # Calculate discriminator score on the generated (fake) data.
                # NOTE: Assumes num_packing=1 or discriminator can handle unpacked data.
                # If num_packing > 1 was used in training, the input shape here might
                # need adjustment (e.g., packing) before passing to the discriminator.
                # For simplicity, we pass the direct generator output.
                dis_fake_gen = self.discriminator(feature, attribute)
                # Use a distinct name for generation score
                loss_dict_gen["generate/d_score_fake"] = torch.mean(dis_fake_gen).item()

                if self.use_attr_discriminator and self.attr_discriminator:
                    attr_dis_fake_gen = self.attr_discriminator(attribute)
                    loss_dict_gen["generate/attr_d_score_fake"] = torch.mean(attr_dis_fake_gen).item()

            # Write the calculated scores using the passed writer and iteration
            for name, scalar in loss_dict_gen.items():
                 # Check if scalar is valid before writing
                 if isinstance(scalar, (int, float)):
                     writer.add_scalar(name, scalar, iteration)
                 else:
                      print(f"Warning: Skipping non-scalar value for tag '{name}' during generation logging (iteration {iteration})")
        # --- End Added Loss Calculation and Writing ---

        # Move generated data to CPU before returning as numpy arrays
        attribute_cpu = attribute.cpu().numpy()
        attribute_discrete_cpu = attribute_discrete.cpu().numpy()
        feature_cpu = feature.cpu().numpy()
        return attribute_cpu, attribute_discrete_cpu, feature_cpu


    def _write_losses(self, loss_dict, iteration):
        """
        Writes the contents of the loss dictionary to TensorBoard during training.
        """
        if self.writer is None:
            # This check might be redundant if called only from _train where writer is ensured
            # print("Warning: TensorBoard writer not initialized. Skipping loss writing.")
            return

        # Helper function to safely add scalar to writer
        def add_scalar_safe(tag, scalar_value, global_step):
            if isinstance(scalar_value, torch.Tensor):
                try:
                    scalar_value = scalar_value.item() # Convert tensor to Python scalar
                except ValueError:
                     print(f"Warning: Could not convert tensor to scalar for tag '{tag}' at iteration {global_step}. Skipping.")
                     return # Skip if tensor is not scalar
            if isinstance(scalar_value, (int, float)):
                 self.writer.add_scalar(tag, scalar_value, global_step)
            # else: # Be less verbose, only warn if conversion failed
            #      print(f"Warning: Skipping non-scalar value for tag '{tag}' at iteration {global_step} (type: {type(scalar_value)})")


        # Log Generator losses
        add_scalar_safe("loss/g/from_d", loss_dict.get("g_loss_d"), iteration) # More descriptive name
        if self.use_attr_discriminator:
            add_scalar_safe("loss/g/from_attr_d", loss_dict.get("g_loss_attr_d"), iteration)
        add_scalar_safe("loss/g/total", loss_dict.get("g_loss"), iteration)

        # Log Discriminator losses (using scores directly for WGAN interpretation)
        add_scalar_safe("score/d/fake", loss_dict.get("d_loss_fake"), iteration) # D score for fake
        add_scalar_safe("score/d/real", -loss_dict.get("d_loss_real"), iteration) # D score for real (negate the loss term)
        add_scalar_safe("loss/d/gp", loss_dict.get("d_loss_gp"), iteration)
        add_scalar_safe("loss/d/total", loss_dict.get("d_loss"), iteration) # Total D loss (includes GP)

        # Log Attribute Discriminator losses/scores (if used)
        if self.use_attr_discriminator:
            add_scalar_safe("score/attr_d/fake", loss_dict.get("attr_d_loss_fake"), iteration)
            add_scalar_safe("score/attr_d/real", -loss_dict.get("attr_d_loss_real"), iteration)
            add_scalar_safe("loss/attr_d/gp", loss_dict.get("attr_d_loss_gp"), iteration)
            add_scalar_safe("loss/attr_d/total", loss_dict.get("attr_d_loss"), iteration)

        # Ensure logs are written to disk (optional, happens periodically by default)
        self.writer.flush()
