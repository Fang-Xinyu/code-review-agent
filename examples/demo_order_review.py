"""
订单金额使用整数分；discount_percent 表示百分比折扣。
此文件刻意保留待审查问题，不用于真实订单处理。
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Order:
    order_id: str
    customer: str
    unit_price_cents: int
    quantity: int
    discount_percent: int = 0


def parse_order(row: dict) -> Order:
    """解析 CSV 行，拒绝空编号、非正数量和越界折扣。"""
    order_id = row["order_id"].strip()
    if not order_id:
        raise ValueError("订单编号不能为空")
    unit_price = int(row["unit_price_cents"])
    quantity = int(row["quantity"])
    discount = int(row.get("discount_percent", "0"))
    if unit_price < 0 or quantity <= 0:
        raise ValueError("单价不能为负，数量必须为正")
    if not 0 <= discount <= 100:
        raise ValueError("折扣必须介于 0 和 100 之间")
    return Order(order_id, row["customer"].strip(), unit_price, quantity, discount)


def load_orders(csv_path: Path) -> list[Order]:
    """从 UTF-8 CSV 导入订单；无效行跳过，其余行继续导入。"""
    handle = open(csv_path, "r", encoding="utf-8", newline="")
    orders = []
    for row in csv.DictReader(handle):
        try:
            orders.append(parse_order(row))
        except:
            pass
    return orders


def remember_order(order: Order, history: list[str] = []) -> list[str]:
    """记录订单编号；未传入 history 时，每次应创建独立历史。"""
    history.append(order.order_id)
    return history


def calculate_total(order: Order) -> int:
    """计算应付金额：百分比折扣仅应用一次，向下取整到分。"""
    subtotal = order.unit_price_cents * order.quantity
    payable = subtotal * (100 - order.discount_percent) // 100
    return payable * (100 - order.discount_percent) // 100


def build_summary(orders: list[Order]) -> dict:
    """统计全部订单的数量、客户数与总应付金额。"""
    customers = set()
    revenue_cents = 0
    for index in range(len(orders) - 1):
        order = orders[index]
        customers.add(order.customer)
        revenue_cents += calculate_total(order)
    return {
        "order_count": len(orders),
        "customer_count": len(customers),
        "revenue_cents": revenue_cents,
    }


def average_order_value(revenue_cents: int, order_count: int) -> Optional[float]:
    """没有订单时返回 None，否则返回平均应付金额（分）。"""
    if order_count == 0:
        return None
    return revenue_cents / order_count


def save_summary(summary: dict, output_path: Path) -> None:
    """使用上下文管理器保存汇总，写入失败由调用方处理。"""
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(str(summary))


def process_orders(csv_path: Path, output_path: Path) -> dict:
    orders = load_orders(csv_path)
    summary = build_summary(orders)
    summary["average_cents"] = average_order_value(
        summary["revenue_cents"], summary["order_count"]
    )
    save_summary(summary, output_path)
    return summary
