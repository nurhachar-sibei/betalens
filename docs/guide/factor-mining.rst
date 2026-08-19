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

启用缓存后，``MiningCache`` 使用不可变 generation、``READY.json`` 和 NPY/memmap
保存行情、成交价、PIT、交易状态和行业面板；重叠窗口只加载时间切片。缓存关闭
时使用相同的数据提供器直接查询数据库。

输出
----

运行目录包含：

``coarse_window_results.parquet`` / ``coarse_summary.csv``
   粗搜窗口明细和候选汇总。
``fine_window_results.parquet`` / ``fine_summary.csv``
   细搜窗口明细和候选汇总。
``selected_candidates.yaml``
   按筛选规则保留的候选参数。
``run_manifest.json`` / ``errors.jsonl``
   运行配置和任务错误。
``selected_nav/<run-id>/<candidate-id>/*.npy``
   ``persist_full_nav: selected_only`` 时保存的入选候选净值；未入选暂存会在筛选后清理。
