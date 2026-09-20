# 把 LIBERO-10 被操作对象替换成 unseen 物体的操作说明

## 目标

在 LIBERO-10 的任务/场景中，把“被操作目标”从 memory 里见过的物体（如 `moka_pot_1`）
替换成 memory 里没有的物体（如 `bbq_sauce_1`、`salad_dressing_1`），
用于测试 pointcloud_action memory 对 **unseen 目标物体** 的检索和动作迁移能力。

---

## 先确认：哪些物体是 unseen

当前 memory 来自 LIBERO-90，里面没有以下物体：

- `bbq_sauce_1`
- `salad_dressing_1`

可以用下面命令检查：

```bash
cd /data1/liu/exp/counterfactual/external/cosmos-policy
python - <<'PY'
import torch, re
mem = torch.load("memory_system/pointcloud_action/pointcloud_action_memory.pt", map_location="cpu", weights_only=False)
items = set()
for r in mem["records"]:
    it = str(r.get("arguments", {}).get("item", ""))
    items.add(re.sub(r"_\d+$", "", it))
print("bbq_sauce in memory:", "bbq_sauce" in items)
print("salad_dressing in memory:", "salad_dressing" in items)
PY
```

---

## 两种做法

### 做法 A：只做“局部 Pick 场景替换”（推荐先做）

pointcloud_action memory 只需要一个能测试 Pick 的场景，不一定要完成完整 LIBERO 任务。
所以最省事的方式是：

1. 选一个 LIBERO-10 的简单桌面场景，例如：
   - `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`
   - 或 `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy`

2. 复制它的 BDDL 文件，例如：

```bash
cp \
  /data1/liu/exp/counterfactual/external/LIBERO-plus/libero/libero/bddl_files/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl \
  /data1/liu/exp/counterfactual/external/LIBERO-plus/libero/libero/bddl_files/libero_10/KITCHEN_SCENE3_pick_up_the_bbq_sauce.bddl
```

3. 修改 BDDL 中的关键部分：

```lisp
;; 原来
(:objects
  moka_pot_1 - moka_pot
  ...
)

;; 改成
(:objects
  bbq_sauce_1 - bbq_sauce
  ...
)
```

同时要改：

- `(:obj_of_interest ...)`
- `(:init ...)` 中物体的放置位置
- `(:goal ...)`，如果只测 Pick，可以简化为：
  ```lisp
  (And (Held bbq_sauce_1))
  ```
  或者保留一个简单的放置目标
- `(:language ...)` 改成：
  ```text
  pick up the bbq sauce
  ```

4. 确认 `bbq_sauce` 这个物体类型已经在 LIBERO-Plus 环境里注册。
   如果是从 LIBERO-Object 里拿的物体，通常已经注册，可以直接用。

5. 为新 BDDL 准备一个 `.pruned_init` 初始状态文件。
   最简单的方式是参考 LIBERO-Object 中同类物体的初始状态，或者用 LIBERO-Plus 的 task generation 工具重新生成。

6. 验证环境能起来：

```python
from memory_system.pointcloud_action.offline.extraction import create_env
env = create_env("KITCHEN_SCENE3_pick_up_the_bbq_sauce", 256, suite="libero_10")
env.reset()
print("ok")
```

7. 然后就可以用现有的 local Pick 评测逻辑跑 memory：
   - 当前场景点云 → memory 检索
   - 得到 ready pose → move to ready
   - 执行 action → 判断是否抓起

---

### 做法 B：完整替换 LIBERO-10 任务

如果希望保留完整任务语义，比如“把 moka pot 放到 stove 上”变成“把 bbq sauce 放到 stove 上”，
那么除了上面 BDDL 的 `(:objects ...)` 外，还要仔细改：

- `(:obj_of_interest ...)`
- `(:init ...)`
- `(:goal ...)`
- `(:language ...)`
- `.pruned_init`
- 可能的 task classification / benchmark 注册

这种改动成本较高，而且 `bbq_sauce` 这类物体不一定有和 `moka_pot` 相同的 affordance/region，
所以容易破坏任务可行性。

---

## 关键注意点

1. **unseen 是相对 memory 而言的**  
   只要 memory 中没有该物体类别，就算 unseen。

2. **替换后不要把这个物体的 demo 加进 memory**  
   否则就不叫 unseen 了。

3. **对 pointcloud_action memory 来说，最重要的是“当前场景里能拿到目标物体的点云和 object frame”**  
   所以不一定要完整任务，局部 Pick 场景就够。

4. **最稳的 unseen 评测仍然是 LIBERO-Object**  
   如果只是想先看结论，直接用：
   - `pick_up_the_bbq_sauce_and_place_it_in_the_basket`
   - `pick_up_the_salad_dressing_and_place_it_in_the_basket`

   这些任务不需要改 BDDL，目标物体本身就已经是 unseen。

---

## 建议路线

1. 先用 LIBERO-Object 的 2 个 unseen 物体跑 20+20 cases，得到 baseline。
2. 如果还需要“LIBERO-10 场景 + unseen 目标 + 干扰物”的叠加测试，
   再按做法 A 生成少量自定义 BDDL/init，跑 local Pick。
3. 不要一开始就做完整任务替换，成本高且容易引入环境错误。
