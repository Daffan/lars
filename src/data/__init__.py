from src.data.data_loader import DataLoader
from src.data.tabmwp import TableMWPDataLoader
from src.data.aqua import AQuADataLoader, ZeroShotCoTDataLoader
from src.data.gsm8k import GSMDataLoader
from src.data.spider import SpiderDataLoader
from src.data.strategy_qa import StrategyQADataLoader, MAWPSDataLoader
from src.data.cogs import COGSDataLoader

data_loader_dict = {
    "tabmwp": TableMWPDataLoader,
    "aqua": AQuADataLoader,
    "gsm8k": GSMDataLoader,
    "zero-shot-cot": ZeroShotCoTDataLoader,
    "spider": SpiderDataLoader,
    "strategy-qa": StrategyQADataLoader,
    "mawps": MAWPSDataLoader,
    "cogs": COGSDataLoader,
}