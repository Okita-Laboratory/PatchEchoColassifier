import torch
from torch.nn import functional as F
from scipy.stats import wasserstein_distance

class DistillationLoss(torch.nn.Module):

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard', 'soft2', 'soft3', 'soft4']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are fed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop through the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
        elif self.distillation_type == 'soft2':
            # Compute Earth Mover's Distance (EMD)
            outputs_prob = F.softmax(outputs_kd, dim=1).cpu().detach().numpy()
            teacher_prob = F.softmax(teacher_outputs, dim=1).cpu().detach().numpy()
            batch_size = outputs_prob.shape[0]
            distillation_loss = 0.0
            for i in range(batch_size):
                distillation_loss += wasserstein_distance(outputs_prob[i], teacher_prob[i])
            distillation_loss /= batch_size
            distillation_loss = torch.tensor(distillation_loss, requires_grad=True, device=outputs_kd.device)
        elif self.distillation_type == 'soft3':
            # Compute Jensen-Shannon Divergence (JS Divergence)
            outputs_prob = F.softmax(outputs_kd, dim=1)
            teacher_prob = F.softmax(teacher_outputs, dim=1)
            mean_prob = 0.5 * (outputs_prob + teacher_prob)
            distillation_loss = 0.5 * (
                F.kl_div(F.log_softmax(outputs_prob, dim=1), mean_prob, reduction='batchmean') +
                F.kl_div(F.log_softmax(teacher_prob, dim=1), mean_prob, reduction='batchmean')
            )
        elif self.distillation_type == 'soft4':
            # Compute Hinge Loss
            outputs_prob = F.softmax(outputs_kd, dim=1)
            teacher_prob = F.softmax(teacher_outputs, dim=1)
            hinge_margin = 1.0
            distillation_loss = torch.mean(
                torch.clamp(hinge_margin - outputs_prob * teacher_prob, min=0.0).pow(2)
            )

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss
