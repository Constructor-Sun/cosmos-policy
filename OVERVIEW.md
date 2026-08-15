数据流

启动 → _load_target_demos() 预计算 (delta_VAE, proprio, scale) 矩阵
  ↓
每个 chunk 边界 → VAE capture → detect_intervention()
  ↓ 检测到 t*
delta_current = flatten(VAE(t*) - VAE(0))
best_idx = argmax(cosine_sim(delta_current, delta_VAE_matrix))
target_proprio, target_scale = proprio[best_idx], scale[best_idx]
offset = (target_ee - current_ee) / target_scale
  ↓
注入（复用现有 _offset_active 分支）”

你认为这个修改合理吗？能够实现自动化修正的需要吗？