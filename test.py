import os
import psutil
import GPUtil
import time
import threading
from datetime import datetime
import pandas as pd

from lightrag import LightRAG, QueryParam
from lightrag.llm import hf_model_complete, hf_embedding, hf_model_complete_batch, initialize_hf_model_batch
from lightrag.utils import EmbeddingFunc, logger, set_logger
from lightrag.config import DEBUG
from transformers import AutoModel, AutoTokenizer

WORKING_DIR = "./json"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

if DEBUG:
    print("-------------IN DEBUGGING MODE-------------\n")
else:
    print("-------------IN RUNNING MODE-------------\n")

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,
    llm_model_func_batch = hf_model_complete_batch,
    llm_model_initial = initialize_hf_model_batch,
    llm_model_name="Qwen/Qwen2.5-7B-Instruct",
    chunk_batch_size = 6,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=5000,
        func=lambda texts: hf_embedding(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embed_model=AutoModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            ),
        ),
    ),
)

def monitor_resources_real_time(process, monitoring_data, interval=1):
    """
    实时监控进程的资源使用情况(CPU, 内存, I/O, GPU)。
    :param process: 要监控的进程对象(psutil.Process())
    :param monitoring_data: 用于存储监控数据的 DataFrame
    :param interval: 监控间隔时间(秒)
    """
    while not stop_event.is_set():
        # 获取 CPU 和内存使用情况
        cpu_percent = process.cpu_percent(interval=interval)
        memory_info = process.memory_info().rss / (1024 * 1024)
        io_counters = process.io_counters()
        io_read_bytes = io_counters.read_bytes
        io_write_bytes = io_counters.write_bytes

        now = datetime.now()
        formatted_time = now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        sys_info_data = {
            "时间": formatted_time,
            "CPU 使用率": cpu_percent,
            "内存使用 (MB)": memory_info,
            "I/O 读取字节数": io_read_bytes,
            "I/O 写入字节数": io_write_bytes
        }

        # 添加每个 GPU 的数据
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_name = f"{gpu.name}-{gpu.uuid}"
            sys_info_data[f"{gpus_id[gpu_name]}-负载 (%)"] = gpu.load * 100
            sys_info_data[f"{gpus_id[gpu_name]}-显存使用 (MB)"] = gpu.memoryUsed
            sys_info_data[f"{gpus_id[gpu_name]}-显存总量 (MB)"] = gpu.memoryTotal
            sys_info_data[f"{gpus_id[gpu_name]}-显存用量 (%)"] = gpu.memoryUsed / gpu.memoryTotal * 100

        monitoring_data.append(sys_info_data)

def monitor_function(func, csv_file_path, *args, **kwargs):
    # 获取当前进程对象
    process = psutil.Process()

    # 定义表头列表
    sys_info_header = [
        "时间", "CPU 使用率", "内存使用 (MB)", "I/O 读取字节数", "I/O 写入字节数"
    ]

    global gpus_id
    gpus_id = {}
    # 为每个 GPU 添加专属的列
    gpus = GPUtil.getGPUs()
    for i, gpu in enumerate(gpus):
        gpu_name = f"{gpu.name}-{gpu.uuid}"
        gpu_simplified_name = f"GPU{i}"
        gpus_id[gpu_name] = gpu_simplified_name
        sys_info_header.extend([
            f"{gpus_id[gpu_name]}-负载 (%)",
            f"{gpus_id[gpu_name]}-显存使用 (MB)",
            f"{gpus_id[gpu_name]}-显存总量 (MB)",
            f"{gpus_id[gpu_name]}-显存用量 (%)"
        ])

    # 用于存储监控数据的列表
    monitoring_data = []

    # 创建停止事件
    global stop_event
    stop_event = threading.Event()

    # 启动监控线程
    monitor_thread = threading.Thread(target=monitor_resources_real_time, args=(process, monitoring_data))
    monitor_thread.start()

    # 执行目标函数
    result = func(*args, **kwargs)

    # 停止监控线程
    stop_event.set()
    monitor_thread.join()

    # 将监控数据转换为 DataFrame
    df = pd.DataFrame(monitoring_data, columns=sys_info_header)

    # 将 DataFrame 写入 CSV 文件
    df.to_csv(csv_file_path, index=False, encoding="utf-8")

    return result

t1 = time.time()
# 执行插入操作并监控资源
with open("./book.txt", "r", encoding="utf-8") as f:
    monitor_function(lambda: rag.insert(f.read()), csv_file_path ="./sys_resource/insert.csv")

t2 = time.time()
print(f"insert time: {t2-t1}")

# 执行查询操作并监控资源
modes = ["naive", "local", "global", "hybrid"]
for mode in modes:
    print(f"{mode} :\n")
    print(monitor_function(lambda: rag.query("What are the top themes in this story?", param=QueryParam(mode=mode)), csv_file_path = f"./sys_resource/{mode}.csv"))
