"""实践一入口：组织两个样本的一次 SGD 更新。"""

from pathlib import Path

from data import prepare_step_data
from models import create_linear_model
from optimizers import create_optimizer
from reporting import save_step_trace
from training import run_single_update


LEARNING_RATE = 0.1
OUTPUT_DIR = Path(__file__).parent / "results"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    x, y = prepare_step_data()
    model = create_linear_model()
    optimizer = create_optimizer(model.parameters(), "SGD", LEARNING_RATE)
    rows = run_single_update(model, optimizer, x, y)
    save_step_trace(rows, OUTPUT_DIR / "step_trace.csv")


if __name__ == "__main__":
    main()
