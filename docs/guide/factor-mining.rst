因子参数挖掘
============

``betalens.factor.mining`` 提供两阶段参数挖掘入口。参数空间、执行模式、
滑动窗口和筛选规则放在 ``parameter_space.yaml``；进程数、缓存和输出目录
放在 ``performance.yaml``。

入口
----

.. code-block:: python

   from betalens.factor.mining import run_mining

   result = run_mining(
       "betalens-factor/alpha101/mining/parameter_space.yaml",
       "betalens-factor/alpha101/mining/performance.yaml",
   )

``result.launch_id`` 标识本次启动；``result.factor_runs`` 为逐因子结果。每个
``FactorMiningResult`` 提供 ``factor_id``、``run_id``、``run_dir``、运行状态、
粗细搜窗口结果、候选汇总和最终赢家。

粗搜使用 Optuna 分布在全局范围内进行稀疏采样；粗搜候选按配置规则汇总后，
引擎自动在优胜候选的邻域生成细粒度网格。Optuna 只负责分布和 trial，
worker 只接收普通参数字典，不负责缓存或进程调度。

因子 hook
---------

因子模块提供 ``make_mining_spec(params) -> MiningSpec``：

* ``precomputed``：完整区间只计算一次因子；没有窗口变换时完整净值只回测一次，
  各窗口表现从净值切片得到。
* ``rolling_fit``：通过 ``fit_window`` 为每个窗口独立拟合并生成权重。
* ``window_transform``：可在不重算因子的情况下做窗口级权重变换，例如 DISP 的
  pretom 过滤。

缓存
----

``cache.data_enabled: true`` 时，每个因子任务在自己的 ``cache/datasets`` 中使用
NPY/memmap 保存行情、成交价、PIT、交易状态和行业面板；重叠窗口只加载时间切片。
``cache/input_manifest.json`` 描述输入缓存，``cache/results.sqlite3`` 增量保存搜索
进度和计算结果。关闭数据缓存时直接查询数据库，但结果数据库仍会生成。

输出
----

每个因子、每次运行使用独立目录：

.. code-block:: text

   outputs/mining/ALPHA3/20260819_220000_a1b2c3d4/
     metadata.yaml
     cache/
       input_manifest.json
       datasets/
       results.sqlite3
     audit/
       挖掘审计.xlsx
       运行日志.log

``metadata.yaml`` 记录配置指纹、环境、Git 状态、运行状态、结果计数和异常。
``挖掘审计.xlsx`` 包含运行概览、参数空间、搜索进度、全部窗口表现、候选汇总、
赢家参数、赢家汇总和错误。旧版扁平 CSV/Parquet、``run_manifest.json``、
``selected_candidates.yaml`` 和 ``selected_nav`` 不再生成。
