import torch

from torch import nn
from torch.nn import Parameter, Conv2d
from torch.nn import functional as F


class OrthogonalConv2d(Conv2d):
    """
    A 2D convolutional layer that enforces orthogonality constraints on its weights.
    
    This layer extends PyTorch's Conv2d to implement the orthogonal linear transformation
    described in the paper's methodology. It supports multiple orthogonalization methods:
    - SVD (Singular Value Decomposition)
    - QR decomposition
    - Eigenvalue decomposition
    
    The layer maintains orthogonality throughout training, which helps preserve spectral
    relationships in the reduced-rank representation while allowing optimization of the
    transformation matrix.

    Args:
        *args: Arguments passed to Conv2d
        initial_weight (torch.Tensor, optional): Initial weight matrix
        ortho_constrain (str, optional): Orthogonalization method to use:
            - "svd": SVD-based orthogonalization
            - "qr": QR decomposition-based orthogonalization
            - "eig": Eigenvalue decomposition-based orthogonalization
            - None: No orthogonalization (standard Conv2d behavior)
        **kwargs: Additional keyword arguments passed to Conv2d
    """

    def __init__(self, *args, initial_weight=None, ortho_constrain=None, **kwargs):
        """Initialize the orthogonal convolutional layer."""
        super(OrthogonalConv2d, self).__init__(*args, **kwargs)
        self.ortho_constrain = ortho_constrain

        # Initialize weights and select appropriate orthogonalization method
        if self.ortho_constrain == "svd":
            self.weight = Parameter(torch.zeros((self.out_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1])),requires_grad=self.weight.requires_grad)
            self.get_constrained_weights = self.svd_constrain_weights
        elif self.ortho_constrain == "qr":
            self.weight = Parameter(torch.zeros((self.out_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1])),requires_grad=self.weight.requires_grad)
            self.get_constrained_weights = self.qr_constrain_weights
        elif self.ortho_constrain == "eig":
            self.weight = Parameter(torch.zeros((self.out_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1])),requires_grad=self.weight.requires_grad)
            self.get_constrained_weights = self.eig_constrain_weights
        else:
            self.get_constrained_weights = lambda: self.weight
        self.reset_parameters(initial_weight)

    def reset_parameters(self, initial_weight=None):
        """
        Initialize or reset the layer parameters.
        
        Uses Kaiming initialization if no initial weights provided.
        
        Args:
            initial_weight (torch.Tensor, optional): Initial weight values
        """

        if initial_weight is None:
            nn.init.kaiming_uniform_(self.weight, a=0.75, mode="fan_out")
        else:
            self.weight.parameter.data = initial_weight

        # If bias is used, initialize it to zero
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def eig_constrain_weights(self):
        """
        Orthogonalize weights using eigenvalue decomposition.
        
        Computes orthogonal basis from eigenvectors of weight matrix
        and projects weights onto this basis.
        
        Returns:
            torch.Tensor: Orthogonalized weight tensor
        """
        weight = self.weight.reshape(self.out_channels,-1)
        _, eigenvectors = torch.linalg.eigh(weight @ weight.T)
        orthogonal_basis = eigenvectors[:, :self.in_channels]
        ortho_weight = orthogonal_basis @ orthogonal_basis.T @ weight
        return ortho_weight.reshape(self.weight.data.shape).type_as(self.weight)

    def svd_constrain_weights(self):
        """
        Orthogonalize weights using SVD.
        
        Note: SVD computation is performed on CPU due to CUDA implementation issues.
        
        Returns:
            torch.Tensor: Orthogonalized weight tensor using right singular vectors
        """
        weight = self.weight.reshape(self.out_channels, -1)
        weight_tensor = weight.cpu() # Move to CPU due to CUDA SVD limitations
        _, _, Vh = torch.linalg.svd(weight_tensor) # Use right singular vectors
        return Vh.mH.reshape(self.weight.data.shape)[:, :self.in_channels].type_as(weight)

    def qr_constrain_weights(self):
        """
        Orthogonalize weights using QR decomposition.
        
        Computes Q matrix from QR decomposition of weight correlation matrix.
        
        Returns:
            torch.Tensor: Orthogonalized weight tensor using Q matrix
        """
        weight = self.weight.reshape(self.out_channels, -1)
        test_tensor = weight@weight.mH # Compute weight correlation matrix
        Q, _ = torch.linalg.qr(test_tensor) # Extract orthogonal factor
        return Q.type_as(weight, device=self.device).unsqueeze(-1).unsqueeze(-1)[:, :self.in_channels, :, :].reshape(self.weight.data.shape)

    def forward(self, inputs):
        """
        Forward pass applying orthogonality constraint if enabled.
        
        Args:
            inputs (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Result of convolution with orthogonalized weights
        """
        if self.ortho_constrain:
            # Apply orthogonality constraint before convolution
            weights = self.get_constrained_weights()
            return F.conv2d(inputs, weights, self.bias, self.stride,
                                        self.padding, self.dilation, self.groups)
        else:
            # Standard convolution without orthogonality constraint
            return F.conv2d(inputs, self.weight, self.bias, self.stride,
                                        self.padding, self.dilation, self.groups)

