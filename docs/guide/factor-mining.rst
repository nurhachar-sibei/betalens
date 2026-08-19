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

Alpha101 自动参数空间
----------------------

Alpha101 可以在 ``parameter_space.yaml`` 中逐因子显式声明参数空间；也可以将
``factors`` 设为 ``all``，由 ``alpha101_parameters.aggregate_mining_factors()``
为 ALPHA1 至 ALPHA101 生成参数定义。显式声明时不会调用自动生成逻辑。

自动生成首先读取 ``alpha101_formulas.py`` 中各 ``alphaN`` 函数的关键字默认值，
再根据参数名后缀分类。``*_window``、``*_lag``、``*_threshold``、
``*_exponent`` 和 ``*_weight`` 可搜索；其他参数固定为论文默认值。默认按照公式
签名顺序最多放开前三个可搜索参数，可通过 ``max_dimensions`` 调整。

.. list-table:: 论文默认值 ``d`` 周围的参考点
   :header-rows: 1

   * - 参数种类
     - 参考点规则
   * - 整数 window / lag
     - 约 ``[0.5d, d, 2d]``，并保证不小于 1
   * - 浮点 window / lag
     - ``[0.5d, d, 2d]``
   * - weight
     - ``[max(0, 0.5d), d, min(1, 1.5d)]``
   * - threshold
     - ``[d-s, d, d+s]``，其中 ``s=max(0.5*abs(d), 0.05)``
   * - exponent
     - ``[0.5d, d, 2d]``
   * - 其他
     - ``[d]``，即固定值

这些参考点只用于计算 ``low`` 和 ``high``，不是离散候选集合。例如 ALPHA3 的
相关系数窗口默认值为 10，自动得到 ``low=5``、``high=20``、``step=1`` 和
``scale=log``；粗搜可以在 5 至 20 的整个整数区间采样。window / lag 使用对数
尺度，其他数值参数使用线性尺度。该规则是围绕论文默认值的启发式边界，不读取
历史数据，也不保证得到统计意义上的最优搜索范围。

参数定义可用字段如下：

* ``type``：``int`` / ``float`` / ``categorical`` / ``bool``；
* ``low`` / ``high``：数值参数的闭区间边界；
* ``step``：正数步长；对数整数只能为 1，对数浮点不能设置；
* ``scale``：``linear`` / ``log``，使用 ``log`` 时 ``low`` 必须大于 0；
* ``choices``：categorical 的非空候选列表；bool 省略时默认为 ``[false, true]``。

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
