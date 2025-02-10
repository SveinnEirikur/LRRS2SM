from einops import rearrange, reduce, repeat
import torch
from torch.nn.functional import mse_loss
from torchmetrics import Metric
from torchmetrics.functional.image import image_gradients

from utilities.common import imgrad_weights, ConvCM, MtMx


class MeanAbsoluteGradientError(Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: bool = False

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent, and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.gradients = image_gradients
        self.add_state('grad_mae', [])

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        pred_dx, pred_dy = image_gradients(preds)
        pred_d = (pred_dx.square() + pred_dy.square()).sqrt()
        target_dx, target_dy = image_gradients(target)
        target_d = (target_dx.square() + target_dy.square()).sqrt()
        # condense into one tensor and avg
        self.grad_mae.append(torch.mean(torch.abs(target_d - pred_d)))

    def compute(self):
        return torch.mean(torch.stack(self.grad_mae))


class SignalReconstructionError_channel_mean(Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: bool = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent, and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self, n_batches=1, n_channels=8, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.add_state('pred_squared_sum', torch.zeros(n_batches, n_channels), dist_reduce_fx='sum')
        self.add_state('diff_squared_sum', torch.zeros(n_batches, n_channels), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        diff = target-preds
        diff[diff==0] += self.eps
        self.pred_squared_sum += reduce(preds**2, '... c h w -> ... c', 'sum')
        self.diff_squared_sum += reduce(diff**2, '... c h w -> ... c', 'sum')

    def compute(self):
        return reduce(10*torch.log10(self.pred_squared_sum/self.diff_squared_sum), '... ->', 'mean')


class SignalReconstructionError(Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: bool = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent, and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self, n_batches=1, n_channels=8, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.add_state('pred_squared_sum', torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('diff_squared_sum', torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        diff = target-preds
        diff[diff==0] += self.eps
        self.pred_squared_sum += reduce(preds**2, '... -> ', 'sum')
        self.diff_squared_sum += reduce(diff**2, '... -> ', 'sum')

    def compute(self):
        return 10*torch.log10(self.pred_squared_sum/self.diff_squared_sum)


class MeanSquaredGradientError(Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: bool = False

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent, and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.gradient_function = image_gradients
        self.add_state('mean_squared_errors', [])


    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        pred_dx, pred_dy = image_gradients(preds)
        pred_d = (pred_dx.square() + pred_dy.square()).sqrt()
        target_dx, target_dy = image_gradients(target)
        target_d = (target_dx.square() + target_dy.square()).sqrt()
        # condense into one tensor and avg
        self.mean_squared_errors.append(torch.mean((target_d - pred_d)**2))

    def compute(self):
        return torch.mean(torch.stack(self.mean_squared_errors))


class WRLoss(Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: bool = False

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent, and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self, rho=1, d=[6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2], sigma=1.0):
        super().__init__()
        self.rho = rho
        self.weight = imgrad_weights
        self.gradient = image_gradients
        self.mse = mse_loss
        self.bands = [idx for idx, val in enumerate(d) if val == 1]
        self.sigma = sigma
        self.loss = 0

    def update(self, image: torch.Tensor, latent: torch.Tensor, alpha: torch.Tensor = torch.Tensor([0.0])):

        w = imgrad_weights(image, bands=self.bands, sigma=self.sigma)
        w = alpha.exp() * w
        lgradh, lgradv = self.gradient(latent)
        lgradm = torch.sqrt(lgradh.square() + lgradv.square())
        self.loss = self.rho*self.mse(w*lgradm, 0.0*lgradm, reduction='sum')

    def compute(self):
        return self.loss


class OldStepCost(Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: bool = False

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent, and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self, d):
        super().__init__()
        self.add_state("costs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.d = d

    def update(self, G: torch.Tensor, MtMBX: torch.Tensor, Y: torch.Tensor,
               W: torch.tensor, fft_of_Dh: torch.Tensor, fft_of_Dv: torch.Tensor,
               tau:torch.Tensor, alpha: torch.Tensor):

        ZhW = W*ConvCM(G,fft_of_Dh.conj())
        ZvW = W*ConvCM(G,fft_of_Dv.conj())
        ZhW = ConvCM(ZhW,fft_of_Dh)
        ZvW = ConvCM(ZvW,fft_of_Dv)
        grad_pen = ZhW + ZvW

        fid_term = reduce(0.5*torch.linalg.vector_norm(MtMx(Y, self.d)-MtMBX, dim=(-2,-1)).square(), "... -> ", "mean")/28750
        pen_term = reduce(tau*alpha.exp()*G*grad_pen, "... -> ", "mean")/1.3

        J = fid_term + pen_term

        self.costs += J
        self.n +=1

    def compute(self):
        return self.costs.float()/self.n.float()


class GStepCost(Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: bool = False

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent, and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("G_costs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, G: torch.Tensor, FBTMTy: torch.Tensor,
               FBtMtMBX: torch.Tensor, W: torch.tensor,
               fft_of_Dh: torch.Tensor, fft_of_Dv: torch.Tensor,
               tau:torch.Tensor, alpha: torch.Tensor):

        ZhW = W*ConvCM(G,fft_of_Dh.conj())
        ZvW = W*ConvCM(G,fft_of_Dv.conj())
        ZhW = ConvCM(ZhW,fft_of_Dh)
        ZvW = ConvCM(ZvW,fft_of_Dv)
        grad_pen = ZhW + ZvW
        gtAtAg = reduce(G*(FBtMtMBX + 2.0*tau*alpha.exp()*grad_pen), "... c h w -> ", "mean")

        J = 0.5*gtAtAg - reduce(G*FBTMTy, "... c h w -> ", "mean")
        with torch.no_grad():
            J += 0.47
        self.G_costs += J
        self.n += 1

    def compute(self):
        return self.G_costs.float()/self.n.float()


class FStepCost(Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: bool = False

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent, and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self, d):
        super().__init__()
        self.add_state("F_costs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.d = d

    def update(self, F: torch.Tensor, G: torch.Tensor, Y: torch.Tensor,
               fft_of_B: torch.Tensor):

        BZT = torch.fft.ifft2(repeat(torch.fft.fft2(G),"b cs h w -> b cs ci h w", ci=Y.shape[-3])*repeat(fft_of_B, "b ci h w -> b cs ci h w", cs=G.shape[-3])).real

        M = torch.zeros([1, 1, F.shape[-2], G.shape[-2], G.shape[-1]]).type_as(G)
        for s in torch.unique(self.d):
            M[:,:, self.d == int(s), ::int(s), ::int(s)] = 1
        MBZT = M*BZT

        J = reduce(0.5 * torch.linalg.vector_norm(rearrange(rearrange(F, "1 ci cs -> ci 1 cs")
                         @ rearrange(MBZT, "b cs ci h w -> ci cs (b h w)"),
                         "c 1 (b h w) -> b c h w", h=MBZT.shape[-2],
                         w=MBZT.shape[-1], b=MBZT.shape[0]) - Y, dim=(-2,-1))**2, "... -> ", "mean")

        self.F_costs += J/10000
        self.n += 1

    def compute(self):
        return self.F_costs.float()/self.n.float()
