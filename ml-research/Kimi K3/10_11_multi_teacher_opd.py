import torch
import torch.nn.functional as F

def multi_teacher_opd_reward(student_logits, teacher_logits, domain_indices, effort_indices, sampled_tokens, clip_threshold):
    """
    Returns: clipped token rewards and selected teacher token log probabilities.
    """
    B, S, V = student_logits.shape
    device = student_logits.device
    dtype = student_logits.dtype
    batch_indices = torch.arange(B, device=device)
    selected_teacher_logits = teacher_logits[domain_indices, effort_indices, batch_indices]
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_token_log_probs = student_log_probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
    teacher_log_probs = F.log_softmax(selected_teacher_logits, dim=-1)
    teacher_token_log_probs = teacher_log_probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
    raw_reward = teacher_token_log_probs - student_token_log_probs.detach()
    rewards = torch.clamp(raw_reward, -clip_threshold, clip_threshold).detach()
    return rewards, teacher_token_log_probs