# Slurm Cluster Skills (Slurm 21.08.5)

這份文件彙整了 Slurm Workload Manager (版本 21.08.5) 的核心概念、日常指令、作業提交工作流、最佳實踐，以及進階的 HDF5 效能分析工具（Profiling）。

---

## 核心概念與架構

Slurm 是一個開源、具備容錯力且高度可擴展的叢集管理與作業排程系統，主要提供三大功能：
1. **資源分配**：為使用者分配計算節點的獨佔或共享存取權限。
2. **執行框架**：提供在已分配節點上啟動、執行與監控作業（通常是平行作業）的框架。
3. **排程仲裁**：透過管理待處理作業（Pending Jobs）的佇列來仲裁資源競爭。

### 系統元件與架構
* **管理節點守护進程 (`slurmctld`)**：中央管理程序，監控資源與排程（通常有備援雙胞胎）。
* **計算節點守护進程 (`slurmd`)**：運行在每個計算節點上，提供容錯的階層式通訊與作業管理。
* **步驟管理進程 (`slurmstepd`)**：由 `slurmd` 啟動，用來管理作業步驟（Job Steps）。

### 資源實體關係
* **Nodes (節點)**：叢集中的計算主機。
* **Partitions (分區)**：將節點分組成的邏輯集合（類似作業佇列），可設定作業大小限制、時間限制、使用者權限等限制。
* **Jobs (作業)**：分配給使用者的特定資源量與時間額度。
* **Job Steps (作業步驟)**：在同一個作業分配（Allocation）內啟動的平行或循序工作集合。

---

## Slurm 核心指令速查與詳細指南

以下是 Slurm 核心指令的快速功能總覽，隨後針對最常用的五個指令 (`sbatch`, `salloc`, `srun`, `squeue`, `sinfo`) 提供詳細的參數、環境變數與用法指南。

### 核心指令功能總覽
* **`sinfo`**：查看分區（Partitions）與節點（Nodes）的狀態與組態。
* **`squeue`**：查看排程佇列中的作業或作業步驟狀態。
* **`scontrol`**：管理員與使用者查詢/修改 Slurm 細部狀態的工具（如 `scontrol show job`）。
* **`srun`**：即時提交作業或在現有分配中啟動平行作業步驟。
* **`sbatch`**：提交批次腳本（Batch Script）至背景排程執行。
* **`salloc`**：即時申請資源分配並開啟一個 Shell（通常在其中使用 `srun` 執行任務）。
* **`scancel`**：取消或發送訊號給作業/作業步驟。
* **`sbcast`**：將檔案從本地傳輸到已分配計算節點的本地暫存磁碟（如 `/tmp`）。
* **`sacct` / `sstat`**：報告作業歷史記帳資訊 / 取得運行中作業的即時資源消耗。

---

## 常用核心指令細部指南

### 1. `sbatch` —— 批次作業提交
用於將寫好的 Shell 腳本提交給 Slurm 進行背景排程。

* **常用資源參數**：
  * `-J, --job-name=<name>`：設定作業名稱。
  * `-p, --partition=<partition>`：指定要在哪個分區（佇列）運行（多個可用逗號分隔）。
  * `-t, --time=<time>`：最長運行時間。格式支援 `minutes`、`minutes:seconds`、`hours:minutes:seconds`、`days-hours`、`days-hours:minutes:seconds`。
  * `-N, --nodes=<minnodes>[-maxnodes]`：請求的節點數量。
  * `-n, --ntasks=<number>`：請求的總任務（MPI Process）數。
  * `-c, --cpus-per-task=<ncpus>`：每個任務分配的 CPU 核心數（常用於 OpenMP/多執行緒作業）。
  * `--mem=<size>`：每節點記憶體限制（如 `10G`、`4000M`）。若指定 `--mem=0` 則會分配該節點上所有的記憶體。
  * `--mem-per-cpu=<size>`：每個 CPU 分配的記憶體上限。
  * `--gres=<list>`：請求通用資源（如 GPU，格式為 `--gres=gpu:4` 或 `--gres=gpu:a100:2`）。
* **檔案輸出與控制參數**：
  * `-o, --output=<pattern>` / `-e, --error=<pattern>`：指定標準輸出與錯誤的檔案名稱格式。常見萬用字元：
    * `%j`：Job ID。
    * `%A`：作業陣列的主作業 ID（Master Job ID）。
    * `%a`：作業陣列中的子任務索引（Array Task Index）。
    * `%x`：作業名稱（Job Name）。
  * `-d, --dependency=<list>`：設定作業間的依賴關係（如 `-d afterok:12345` 代表在 12345 作業成功完成後才可啟動）。
  * `-a, --array=<indexes>`：提交作業陣列（例如 `--array=0-15`，或是 `--array=0-15%4` 限制最多 4 個同時運行）。
  * `--exclusive`：獨佔分配到的節點，不與其他使用者的作業共享。
  * `--mail-type=<type>` / `--mail-user=<email>`：作業狀態改變時發送郵件（如 `BEGIN,END,FAIL,ALL`）。
* **重要環境變數**：在提交的批次腳本中，Slurm 會自動注入以下環境變數供程式碼使用：
  * `SLURM_JOB_ID`：當前作業的 Job ID。
  * `SLURM_SUBMIT_DIR`：提交作業時的路徑。
  * `SLURM_ARRAY_TASK_ID`：作業陣列中當前任務的索引。

### 2. `salloc` —— 互動式資源申請
向排程器即時申請一組資源分配，獲得後預設會開啟一個 Shell，在該 Shell 退出後資源會自動釋放。常用於偵錯與開發。

* **常用參數**：
  * 其資源申請參數與 `sbatch` 完全一致（如 `-N`, `-n`, `-c`, `-t`, `-p`, `--mem`, `--gres` 等）。
* **使用模式**：
  * 申請資源：`salloc -N1 -p debug -t 30:00 bash`
  * 獲得 Shell 後，使用 `srun` 執行具體命令：`srun ./my_program`
  * 結束工作：`exit`（釋放資源）。

### 3. `srun` —— 平行任務啟動
在 Slurm 分配的資源內啟動並行任務（Job Steps）。如果在無分配的環境下直接運行，會自動先隱式向排程器申請資源再執行。

* **常用參數**：
  * `-l, --label`：在每一行 stdout/stderr 輸出前加上任務的 Rank 索引（如 `0: hostname`），便於平行輸出排錯。
  * `-u, --unbuffered`：使輸出不進行緩衝，即時輸出至终端。
  * `-I, --immediate[=seconds]`：如果無法立即獲得所需資源，則直接退出。
  * `--mpi=<type>`：指定 MPI 整合插件（如 `pmix`、`pmi2`），由 Slurm 負責處理平行通訊。
  * `--exclusive`：排他性地將核心分配給此步驟，避免同作業內的其他 `srun` 步驟共享相同的 CPU。
  * `--overlap`：允許此步驟與同作業內的其他步驟共享 CPU 資源。

### 4. `squeue` —— 佇列狀態查詢
檢視排程佇列中所有作業的狀態與排程原因。

* **常用參數**：
  * `-u, --user=<user_list>`：僅顯示指定使用者的作業（多個可用逗號分隔）。
  * `-p, --partition=<partition_list>`：僅顯示特定分區的作業。
  * `-t, --states=<state_list>`：僅顯示特定狀態的作業（常見：`PD` Pending 擱置、`R` Running 運行中、`CG` Completing 結束中）。
  * `-j, --jobs=<job_list>`：查詢指定的 Job ID。
  * `-i, --iterate=<seconds>`：設定每隔幾秒自動重新整理顯示一次狀態。
  * `-S, --sort=<sort_list>`：排序規則（如 `-S P,t` 代表先依分區排序，再依狀態排序）。
  * `-o, --format=<format>`：自訂輸出格式。常見的格式化字元：
    * `%i`：Job ID（若有 job step 則顯示 Step ID）
    * `%j`：作業名稱（Job Name）
    * `%u`：使用者名稱
    * `%t` / `%T`：作業狀態縮寫 / 完整狀態名稱（如 `R` / `RUNNING`）
    * `%M`：已執行時間（格式：`days-hours:minutes:seconds`）
    * `%l`：最長執行時間限制（Time Limit）
    * `%D`：分配的節點數量
    * `%R`：若作業擱置，顯示原因（Reason）；若在執行，顯示節點列表（Nodelist）
    * `%q`：作業的服務品質（QOS）

### 5. `sinfo` —— 分區與節點狀態查詢
檢視計算節點和分區（佇列）的整體可用資源與實體狀態。

* **常用參數**：
  * `-p, --partition=<partition>`：僅查詢指定分區的節點。
  * `-N, --Node`：以單一計算節點為單位列出狀態（預設會把相同分區與相同狀態的節點合併顯示為一行）。
  * `-s, --summarize`：只顯示每個分區的摘要統計。
  * `-R, --responding`：顯示無回應（Down/Drained）的節點清單及其異常原因。
  * `-o, --format=<format>`：自訂輸出格式。常見的格式化字元：
    * `%P`：分區名稱（預設分區名稱後方會帶有 `*`）
    * `%a`：分區是否可用（`up`/`down`）
    * `%l`：分區的作業時間限制
    * `%D`：該行代表的節點數量
    * `%F`：節點統計（格式為 `Allocated/Idle/Other/Total`，例如 `2/3/1/6`）
    * `%N`：節點列表（Node List）
    * `%T`：節點目前狀態（如 `allocated`、`idle`、`down*`、`drain*` 等，有 `*` 代表節點無回應）

---


## 常用作業流程與範例

### 1. 互動式資源申請與任務執行 (`salloc`)
適合開發調試或執行快速測試：
```bash
# 1. 申請 1024 個節點並開啟 bash
salloc -N1024 bash

# 2. 將執行檔廣播分發到所有分配節點的 /tmp 目錄，減少 NFS 負荷
sbcast a.out /tmp/joe.a.out

# 3. 在所有節點上並行執行該程式
srun /tmp/joe.a.out

# 4. 清理並退出分配
srun rm /tmp/joe.a.out
exit
```

### 2. 批次作業腳本提交 (`sbatch`)
在背景排程執行，腳本內部可使用 `#SBATCH` 宣告預設引數：
```bash
# 提交作業並指定節點數、主機與輸出檔案
sbatch -n4 -w "adev[9-10]" -o my.stdout my.script
```
批次腳本 `my.script` 範例：
```bash
#!/bin/sh
#SBATCH --time=00:10:00   # 限時 10 分鐘

/bin/hostname            # 在主節點執行
srun -l /bin/hostname    # 在分配的所有處理器上並行執行，印出任務編號
srun -l /bin/pwd         # 執行下一個步驟
```

### 3. 直接並行執行 (`srun`)
```bash
# 在 3 個節點上各執行一個任務，並顯示任務編號
srun -N3 -l /bin/hostname

# 啟動 4 個任務（不限節點數），顯示任務編號
srun -n4 -l /bin/hostname
```

---

## 大型作業與 MPI 最佳實踐

### 高通量最佳實踐
* **多步驟整合**：若有大量相關的小型任務，建議將它們整合進單一 Slurm 作業中，改用多個 **Job Steps**（作業步驟）循序或並行執行。這比提交大量獨立的 Job 效率更高，對系統負載也小很多。
* **作業陣列 (Job Arrays)**：對於資源需求相同但輸入不同的批次任務集合，使用 Job Arrays 可大幅提升排程與管理效率（可透過單一指令取消或管理整個陣列）。

### MPI 執行模式
根據 MPI 實作不同，主要有三種整合模式：
1. **直接啟動（推薦）**：Slurm 直接啟動任務，並透過 `PMI2` 或 `PMIx` API 初始化通訊（大多數現代 MPI 支援）。
2. **Slurm 資源分配 + mpirun 啟動**：Slurm 建立資源分配，然後由 `mpirun` 使用 Slurm 的基礎建設啟動任務（如舊版 OpenMPI）。
3. **Slurm 資源分配 + 外部啟動**：Slurm 分配資源後，`mpirun` 透過 SSH/RSH 等外部機制啟動任務（此時任務在 Slurm 監控之外，強烈建議配置 `pam_slurm_adopt` 與作業結束時的 Epilog 清理）。

---

## 進階：使用 HDF5 進行作業效能分析 (Profiling)

當需要比一般資料庫更詳細的作業效能指標時，可以使用 `acct_gather_profile/hdf5` 插件。它會定期採樣節點與任務的效能數據，記錄為 **時間序列 (Time Series)** 並累計作業總量。

### 1. 採樣指標分類
* **Energy (能源)**：藉由 `acct_gather_energy/ipmi` 收集節點功耗與 CPU 頻率。
* **Filesystem (檔案系統)**：監控並行檔案系統（如 Lustre）的讀寫次數與傳輸量。
* **Network (網路/互連)**：監控 InfiniBand 等介面的封包與流量。
* **Task (任務效能)**：監控個別任務的 CPU 時間、CPU 使用率、RSS 記憶體、虛擬記憶體大小、頁面錯誤、以及本地磁碟讀寫量。

### 2. 系統配置說明

#### `slurm.conf` 設定
```ini
# 啟用 HDF5 效能分析插件
AcctGatherProfileType=acct_gather_profile/hdf5

# 設定預設採樣頻率（單位：秒）
JobAcctGatherFrequency=30
```

#### `acct_gather.conf` 設定
```ini
# 必須指定所有計算節點皆能存取的共享資料夾路徑
ProfileHDF5Dir=/path/to/shared/profile_dir

# （選用）預設收集類型（僅建議測試環境使用，否則會產生海量檔案）
# ProfileHDF5Default=all
```

### 3. 使用與分析指令

#### 啟動 Profiling
使用 `salloc`, `sbatch` 或 `srun` 時，透過 `--profile` 與 `--acctg-freq` 參數控制收集：
```bash
# 啟用所有指標收集，並自訂各指標的採樣頻率（單位：秒）
srun --profile=all --acctg-freq task=10,energy=5,filesystem=10,network=10 ./my_program
```
* `--profile` 可選值：`all`, `none`, 或逗號分隔的列表（例如 `energy,task`）。

#### 合併資料 (`sh5util`)
作業運行時，各節點會將步驟效能寫入共享目錄的臨時節點文件中。作業結束後，需使用 `sh5util` 工具將這些破碎的節點步驟檔案合併成單一的 HDF5 檔案：
```bash
# 合併特定作業的效能數據
sh5util -j $SLURM_JOB_ID

# 在批次腳本中，常以下列方式提交合併任務（在主作業結束後自動執行）
sbatch -n1 -d$SLURM_JOB_ID --wrap="sh5util -j $SLURM_JOB_ID"
```

#### 數據導出與視覺化
* **導出 CSV**：`sh5util` 可將 HDF5 數據導出為 CSV 格式以利試算表或分析腳本讀取。
* **視覺化工具**：推薦使用 HDF5 官方的 **HDFView** 工具直接打開 `.h5` 檔案，可階層式地瀏覽 `Steps -> Nodes/Tasks -> Time Series / Totals` 的詳細圖表與屬性。

