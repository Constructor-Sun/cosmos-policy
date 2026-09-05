# Plus-Only 物体加载异常现象

## 现象

在 `libero_10` 的 `counterfactual_plusonly_*` 单物体场景中，oracle 点云提取结果完全错误。

例如：

```text
bottle_of_alfredo_sauce
bottle_of_antihistamines
bottle_of_aspirin
bottle_of_baby_oil
bottle_of_barbecue_sauce__2
```

点云不是桌面上的物体，而是 gripper 附近的错误几何。

## 原因

这些 plus-only 物体的 MJCF 中，内部 body 名都是：

```xml
<body name="object" ...>
```

加载进 LIBERO/MuJoCo 后，实际结果变成：

- 存在一个自由 body：`<object>_1_main`
- 但该物体的 visual/collision geom 被挂到了：
  `gripper0_leftfinger`

因此：

- 物体并没有作为桌面上的自由物体正确存在；
- oracle 点云提取忠实提取了 MuJoCo 中名字匹配的 geom，但那些 geom 本身就在 gripper 上；
- 所以点云看起来完全不对。

## 证据

- `complete_point_cloud()` 提取出的 bbox 非常小且位于 gripper 附近；
- 检查 MuJoCo body/geom 可以看到 geom 的 parent body 是 `gripper0_leftfinger`；
- 更早的 `plus_only_object_replacement` 结果：100 个 case，success = 0。

## 结论

这不是点云提取算法的问题，而是这些 plus-only 物体的 MJCF/robosuite 加载结果有问题。

在这些物体被正确加载为自由物体之前，不应把它们作为有效的单物体 Pick 测试对象。
