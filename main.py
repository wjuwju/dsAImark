import os
import time
import schedule
from openai import OpenAI
import ccxt
import pandas as pd
from datetime import datetime
import json
import re
from dotenv import load_dotenv
import logging

load_dotenv()

# 配置日志系统
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 创建日志文件名（按日期）
log_filename = os.path.join(log_dir, f"deepseek_chat_{datetime.now().strftime('%Y%m%d')}.log")

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

logger = logging.getLogger(__name__)

# 加载配置文件
def load_config():
    """加载配置文件"""
    config_file = "config.json"
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"已从 {config_file} 加载配置")
                return config
        else:
            logger.warning(f"配置文件 {config_file} 不存在，使用默认配置")
            return None
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}，使用默认配置")
        return None

# 初始化DeepSeek客户端
deepseek_client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

# 初始化OKX交易所（用于交易）
exchange = ccxt.okx({
    'apiKey': os.getenv('OKX_API_KEY'),
    'secret': os.getenv('OKX_SECRET'),
    'password': os.getenv('OKX_PASSWORD'),  # OKX需要交易密码
    'options': {
        'defaultType': 'swap',  # OKX使用swap表示永续合约
    },
})

# 创建一个不带认证的公开 exchange 实例（用于获取市场数据）
exchange_public = ccxt.okx({
    'options': {
        'defaultType': 'swap',
    },
})

# 交易参数配置 - 优先从配置文件加载，否则使用默认配置
config_from_file = load_config()
if config_from_file:
    TRADE_CONFIG = config_from_file
else:
    # 默认配置
    TRADE_CONFIG = {
        'symbol': 'BTC/USDT:USDT',  # OKX的合约符号格式
        'amount': 0.01,  # 交易数量 (BTC)
        'leverage': 10,  # 杠杆倍数
        'timeframe': '15m',  # 使用15分钟K线
        'test_mode': True,  # 测试模式
        'data_points': 96,  # 24小时数据（96根15分钟K线）
        'kline_display_count': 5,  # 显示的K线数量，默认5根
        'analysis_periods': {
            'short_term': 20,  # 短期均线
            'medium_term': 50,  # 中期均线
            'long_term': 96  # 长期趋势
        },
        'execution_interval': 60,  # 执行间隔时间（秒），默认60秒
        'retry_interval': 300  # 出错后重试间隔（秒），默认5分钟
    }

# 说明：不使用 OKX 模拟盘标记
# OKX 的模拟盘 API 功能严重受限，几乎所有接口都不可用
# 改为使用实盘 API 读取数据，通过 test_mode 控制是否实际下单
if TRADE_CONFIG.get('test_mode', True):
    logger.info("测试模式：只分析不下单")
    if TRADE_CONFIG.get('virtual_mode', False):
        print(f"✅ 虚拟仓位模式：使用虚拟金额 ${TRADE_CONFIG.get('virtual_balance', 10000):,.2f} 进行模拟交易")
    else:
        print("✅ 测试模式：程序会分析市场并生成信号，但不会实际下单")
else:
    logger.warning("⚠️ 实盘模式：程序会真实下单，请谨慎！")
    print("⚠️ 实盘模式：程序会真实下单，请谨慎！")

# 全局变量存储历史数据
price_history = []
signal_history = []
position = None

# 虚拟仓位管理
virtual_position = None
virtual_balance = TRADE_CONFIG.get('virtual_balance', 10000)
virtual_initial_balance = virtual_balance  # 记录初始金额用于计算总收益


def setup_exchange():
    """设置交易所参数"""
    try:
        # 在实盘模式下设置杠杆
        if not TRADE_CONFIG.get('test_mode', True):
            exchange.set_leverage(
                TRADE_CONFIG['leverage'],
                TRADE_CONFIG['symbol']
            )
            print(f"设置杠杆倍数: {TRADE_CONFIG['leverage']}x")
            logger.info(f"设置杠杆倍数: {TRADE_CONFIG['leverage']}x")
        else:
            print(f"测试模式：跳过杠杆设置")
            logger.info(f"测试模式：跳过杠杆设置")

        # 获取余额
        try:
            balance = exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            print(f"当前USDT余额: {usdt_balance:.2f}")
            logger.info(f"当前USDT余额: {usdt_balance:.2f}")
        except Exception as e:
            logger.warning(f"获取余额失败: {e}")
            if TRADE_CONFIG.get('test_mode', True):
                print(f"测试模式：跳过余额查询")
            else:
                raise

        return True
    except Exception as e:
        error_msg = f"交易所设置失败: {e}"
        print(error_msg)
        logger.error(error_msg)
        
        # 测试模式下即使失败也继续运行
        if TRADE_CONFIG.get('test_mode', True):
            warning_msg = "⚠️ 测试模式：部分初始化失败，但继续运行"
            print(warning_msg)
            logger.warning(warning_msg)
            return True
        
        return False


def calculate_technical_indicators(df):
    """计算技术指标 - 来自第一个策略"""
    try:
        # 移动平均线
        df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()

        # 指数移动平均线
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # 相对强弱指数 (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 布林带
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # 成交量均线
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # 支撑阻力位
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()

        # 填充NaN值
        df = df.bfill().ffill()

        return df
    except Exception as e:
        print(f"技术指标计算失败: {e}")
        return df


def get_support_resistance_levels(df, lookback=20):
    """计算支撑阻力位"""
    try:
        recent_high = df['high'].tail(lookback).max()
        recent_low = df['low'].tail(lookback).min()
        current_price = df['close'].iloc[-1]

        resistance_level = recent_high
        support_level = recent_low

        # 动态支撑阻力（基于布林带）
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]

        return {
            'static_resistance': resistance_level,
            'static_support': support_level,
            'dynamic_resistance': bb_upper,
            'dynamic_support': bb_lower,
            'price_vs_resistance': ((resistance_level - current_price) / current_price) * 100,
            'price_vs_support': ((current_price - support_level) / support_level) * 100
        }
    except Exception as e:
        print(f"支撑阻力计算失败: {e}")
        return {}


def get_market_trend(df):
    """判断市场趋势"""
    try:
        current_price = df['close'].iloc[-1]

        # 多时间框架趋势分析
        trend_short = "上涨" if current_price > df['sma_20'].iloc[-1] else "下跌"
        trend_medium = "上涨" if current_price > df['sma_50'].iloc[-1] else "下跌"

        # MACD趋势
        macd_trend = "bullish" if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else "bearish"

        # 综合趋势判断
        if trend_short == "上涨" and trend_medium == "上涨":
            overall_trend = "强势上涨"
        elif trend_short == "下跌" and trend_medium == "下跌":
            overall_trend = "强势下跌"
        else:
            overall_trend = "震荡整理"

        return {
            'short_term': trend_short,
            'medium_term': trend_medium,
            'macd': macd_trend,
            'overall': overall_trend,
            'rsi_level': df['rsi'].iloc[-1]
        }
    except Exception as e:
        print(f"趋势分析失败: {e}")
        return {}


def get_btc_ohlcv_enhanced():
    """增强版：获取BTC K线数据并计算技术指标"""
    try:
        print(f"🔍 正在获取 {TRADE_CONFIG['symbol']} 的K线数据...")
        # 使用公开 API 获取K线数据（不需要认证，不受模拟盘限制）
        ohlcv = exchange_public.fetch_ohlcv(TRADE_CONFIG['symbol'], TRADE_CONFIG['timeframe'],
                                     limit=TRADE_CONFIG['data_points'])
        
        if not ohlcv or len(ohlcv) == 0:
            print("❌ 获取K线数据为空")
            return None
            
        print(f"✅ 成功获取 {len(ohlcv)} 根K线数据")

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        print(f"📊 DataFrame形状: {df.shape}")
        print(f"📊 最新价格: {df['close'].iloc[-1]:.2f}")

        # 计算技术指标
        print("🔧 正在计算技术指标...")
        df = calculate_technical_indicators(df)

        current_data = df.iloc[-1]
        previous_data = df.iloc[-2]

        # 获取技术分析数据
        print("📈 正在分析市场趋势...")
        trend_analysis = get_market_trend(df)
        if not trend_analysis:
            trend_analysis = {}
            
        print("🎯 正在计算支撑阻力位...")
        levels_analysis = get_support_resistance_levels(df)
        if not levels_analysis:
            levels_analysis = {}
        
        print("✅ 技术分析完成")
        return {
            'price': current_data['close'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'high': current_data['high'],
            'low': current_data['low'],
            'volume': current_data['volume'],
            'timeframe': TRADE_CONFIG['timeframe'],
            'price_change': ((current_data['close'] - previous_data['close']) / previous_data['close']) * 100,
            'kline_data': df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(10).to_dict('records'),
            'technical_data': {
                'sma_5': current_data.get('sma_5', 0),
                'sma_20': current_data.get('sma_20', 0),
                'sma_50': current_data.get('sma_50', 0),
                'rsi': current_data.get('rsi', 0),
                'macd': current_data.get('macd', 0),
                'macd_signal': current_data.get('macd_signal', 0),
                'macd_histogram': current_data.get('macd_histogram', 0),
                'bb_upper': current_data.get('bb_upper', 0),
                'bb_lower': current_data.get('bb_lower', 0),
                'bb_position': current_data.get('bb_position', 0),
                'volume_ratio': current_data.get('volume_ratio', 0)
            },
            'trend_analysis': trend_analysis,
            'levels_analysis': levels_analysis,
            'full_data': df
        }
    except Exception as e:
        print(f"获取增强K线数据失败: {e}")
        return None


def generate_technical_analysis_text(price_data):
    """生成技术分析文本"""
    if not price_data or 'technical_data' not in price_data:
        return "技术指标数据不可用"

    tech = price_data['technical_data']
    trend = price_data.get('trend_analysis', {})
    levels = price_data.get('levels_analysis', {})

    # 检查数据有效性
    def safe_float(value, default=0):
        return float(value) if value and pd.notna(value) else default

    analysis_text = f"""
    【技术指标分析】
    📈 移动平均线:
    - 5周期: {safe_float(tech['sma_5']):.2f} | 价格相对: {(price_data['price'] - safe_float(tech['sma_5'])) / safe_float(tech['sma_5']) * 100:+.2f}%
    - 20周期: {safe_float(tech['sma_20']):.2f} | 价格相对: {(price_data['price'] - safe_float(tech['sma_20'])) / safe_float(tech['sma_20']) * 100:+.2f}%
    - 50周期: {safe_float(tech['sma_50']):.2f} | 价格相对: {(price_data['price'] - safe_float(tech['sma_50'])) / safe_float(tech['sma_50']) * 100:+.2f}%

    🎯 趋势分析:
    - 短期趋势: {trend.get('short_term', 'N/A')}
    - 中期趋势: {trend.get('medium_term', 'N/A')}
    - 整体趋势: {trend.get('overall', 'N/A')}
    - MACD方向: {trend.get('macd', 'N/A')}

    📊 动量指标:
    - RSI: {safe_float(tech['rsi']):.2f} ({'超买' if safe_float(tech['rsi']) > 70 else '超卖' if safe_float(tech['rsi']) < 30 else '中性'})
    - MACD: {safe_float(tech['macd']):.4f}
    - 信号线: {safe_float(tech['macd_signal']):.4f}

    🎚️ 布林带位置: {safe_float(tech['bb_position']):.2%} ({'上部' if safe_float(tech['bb_position']) > 0.7 else '下部' if safe_float(tech['bb_position']) < 0.3 else '中部'})

    💰 关键水平:
    - 静态阻力: {safe_float(levels.get('static_resistance', 0)):.2f}
    - 静态支撑: {safe_float(levels.get('static_support', 0)):.2f}
    """
    return analysis_text


def get_current_position():
    """获取当前持仓情况 - OKX版本（支持虚拟仓位）"""
    global virtual_position
    
    # 如果启用虚拟仓位模式
    if TRADE_CONFIG.get('test_mode', True) and TRADE_CONFIG.get('virtual_mode', False):
        return virtual_position
    
    try:
        positions = exchange.fetch_positions([TRADE_CONFIG['symbol']])

        for pos in positions:
            if pos['symbol'] == TRADE_CONFIG['symbol']:
                contracts = float(pos['contracts']) if pos['contracts'] else 0

                if contracts > 0:
                    return {
                        'side': pos['side'],  # 'long' or 'short'
                        'size': contracts,
                        'entry_price': float(pos['entryPrice']) if pos['entryPrice'] else 0,
                        'unrealized_pnl': float(pos['unrealizedPnl']) if pos['unrealizedPnl'] else 0,
                        'leverage': float(pos['leverage']) if pos['leverage'] else TRADE_CONFIG['leverage'],
                        'symbol': pos['symbol']
                    }

        return None

    except Exception as e:
        print(f"获取持仓失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def safe_json_parse(json_str):
    """安全解析JSON，处理格式不规范的情况"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # 修复常见的JSON格式问题
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON解析失败，原始内容: {json_str}")
            print(f"错误详情: {e}")
            return None


def create_fallback_signal(price_data):
    """创建备用交易信号"""
    # 🔴 修复：处理 price_data 为空的情况
    if not price_data or not isinstance(price_data, dict):
        price_data = {'price': 0}
    
    return {
        "signal": "HOLD",
        "reason": "因技术分析暂时不可用，采取保守策略",
        "stop_loss": price_data['price'] * 0.98,  # -2%
        "take_profit": price_data['price'] * 1.02,  # +2%
        "confidence": "LOW",
        "is_fallback": True
    }


def safe_get_value(data, key, default=None):
    """安全获取字典值，防止NoneType错误"""
    try:
        if data is None:
            return default
        if isinstance(data, dict):
            return data.get(key, default)
        return default
    except Exception as e:
        print(f"安全获取值失败: {e}")
        return default


def analyze_with_deepseek(price_data):
    """使用DeepSeek分析市场并生成交易信号（增强版）"""

    # 🔴 修复：添加空值检查
    if not price_data or not isinstance(price_data, dict):
        print("❌ price_data 为空或无效，使用备用信号")
        return create_fallback_signal({'price': 0})

    # 生成技术分析文本
    technical_analysis = generate_technical_analysis_text(price_data)

    # 获取K线显示数量配置（默认5根）
    kline_count = TRADE_CONFIG.get('kline_display_count', 5)
    
    # 构建K线数据文本
    kline_text = f"【最近{kline_count}根{TRADE_CONFIG['timeframe']}K线数据】\n"
    
    # 🔴 修复：检查 kline_data 是否存在且不为空
    if 'kline_data' in price_data and price_data['kline_data'] is not None:
        kline_data = price_data['kline_data']
        if isinstance(kline_data, list) and len(kline_data) > 0:
            # 使用配置的数量，取最后 kline_count 根K线
            for i, kline in enumerate(kline_data[-kline_count:]):
                if isinstance(kline, dict) and 'close' in kline and 'open' in kline:
                    trend = "阳线" if kline['close'] > kline['open'] else "阴线"
                    change = ((kline['close'] - kline['open']) / kline['open']) * 100
                    kline_text += f"K线{i + 1}: {trend} 开盘:{kline['open']:.2f} 收盘:{kline['close']:.2f} 涨跌:{change:+.2f}%\n"
                else:
                    kline_text += f"K线{i + 1}: 数据格式错误\n"
        else:
            kline_text += "K线数据为空\n"
    else:
        kline_text += "K线数据不可用\n"

    # 添加上次交易信号
    signal_text = ""
    if signal_history and len(signal_history) > 0:
        last_signal = signal_history[-1]
        if isinstance(last_signal, dict):
            signal_text = f"\n【上次交易信号】\n信号: {last_signal.get('signal', 'N/A')}\n信心: {last_signal.get('confidence', 'N/A')}"
        else:
            signal_text = "\n【上次交易信号】\n数据格式错误"

    # 添加当前持仓信息
    current_pos = get_current_position()
    position_text = "无持仓" if not current_pos else f"{current_pos['side']}仓, 数量: {current_pos['size']}, 盈亏: {current_pos['unrealized_pnl']:.2f}USDT"

    prompt = f"""
    你是一个专业的加密货币交易分析师。请基于以下BTC/USDT {TRADE_CONFIG['timeframe']}周期数据进行分析：

    {kline_text}

    {technical_analysis}

    {signal_text}

    【当前行情】
    - 当前价格: ${price_data['price']:,.2f}
    - 时间: {price_data['timestamp']}
    - 本K线最高: ${price_data['high']:,.2f}
    - 本K线最低: ${price_data['low']:,.2f}
    - 本K线成交量: {price_data['volume']:.2f} BTC
    - 价格变化: {price_data['price_change']:+.2f}%
    - 当前持仓: {position_text}
    - 持仓盈亏: {(current_pos['unrealized_pnl'] if current_pos else 0):.2f} USDT

    【防频繁交易重要原则】
    1. **趋势持续性优先**: 不要因单根K线或短期波动改变整体趋势判断
    2. **持仓稳定性**: 除非趋势明确强烈反转，否则保持现有持仓方向
    3. **反转确认**: 需要至少2-3个技术指标同时确认趋势反转才改变信号
    4. **成本意识**: 减少不必要的仓位调整，每次交易都有成本

    【交易指导原则 - 必须遵守】
    1. **趋势跟随**: 明确趋势出现时立即行动，不要过度等待
    2. 因为做的是btc，做多权重可以大一点点
    3. **信号明确性**:
    - 强势上涨趋势 → BUY信号
    - 强势下跌趋势 → SELL信号  
    - 仅在窄幅震荡、无明确方向时 → HOLD信号
    4. **技术指标权重**:
    - 趋势(均线排列) > RSI > MACD > 布林带
    - 价格突破关键支撑/阻力位是重要信号

    【当前技术状况分析】
    - 整体趋势: {price_data.get('trend_analysis', {}).get('overall', 'N/A') if price_data.get('trend_analysis') else 'N/A'}
    - 短期趋势: {price_data.get('trend_analysis', {}).get('short_term', 'N/A') if price_data.get('trend_analysis') else 'N/A'} 
    - RSI状态: {(price_data.get('technical_data', {}).get('rsi', 0) if price_data.get('technical_data') else 0):.1f} ({'超买' if (price_data.get('technical_data', {}).get('rsi', 0) if price_data.get('technical_data') else 0) > 70 else '超卖' if (price_data.get('technical_data', {}).get('rsi', 0) if price_data.get('technical_data') else 0) < 30 else '中性'})
    - MACD方向: {price_data.get('trend_analysis', {}).get('macd', 'N/A') if price_data.get('trend_analysis') else 'N/A'}

    【分析要求】
    基于以上分析，请给出明确的交易信号

    请用以下JSON格式回复：
    {{
        "signal": "BUY|SELL|HOLD",
        "reason": "简要分析理由(包含趋势判断和技术依据)",
        "stop_loss": 具体价格,
        "take_profit": 具体价格, 
        "confidence": "HIGH|MEDIUM|LOW"
    }}
    """

    try:
        # 记录发送给 DeepSeek 的提示词
        system_message = f"您是一位专业的交易员，专注于{TRADE_CONFIG['timeframe']}周期趋势分析。请结合K线形态和技术指标做出判断，并严格遵循JSON格式要求。"
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.1
        )

        # 安全解析JSON
        if not response or not hasattr(response, 'choices') or not response.choices or len(response.choices) == 0:
            error_msg = "❌ DeepSeek API 响应为空或格式错误"
            print(error_msg)
            logger.error(error_msg)
            return create_fallback_signal(price_data)
            
        if not hasattr(response.choices[0], 'message') or not response.choices[0].message:
            error_msg = "❌ DeepSeek API 响应消息为空"
            print(error_msg)
            logger.error(error_msg)
            return create_fallback_signal(price_data)
            
        result = response.choices[0].message.content
        if not result:
            error_msg = "❌ DeepSeek API 响应内容为空"
            print(error_msg)
            logger.error(error_msg)
            return create_fallback_signal(price_data)
        
        # 记录 DeepSeek 的响应
        logger.info("=" * 80)
        logger.info("DeepSeek 的响应:")
        logger.info(result)
        logger.info("=" * 80)
        
        print(f"DeepSeek原始回复: {result}")

        # 提取JSON部分
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1

        if start_idx != -1 and end_idx != 0:
            json_str = result[start_idx:end_idx]
            signal_data = safe_json_parse(json_str)

            if signal_data is None:
                signal_data = create_fallback_signal(price_data)
        else:
            signal_data = create_fallback_signal(price_data)

        # 验证必需字段
        required_fields = ['signal', 'reason', 'stop_loss', 'take_profit', 'confidence']
        if not all(field in signal_data for field in required_fields):
            signal_data = create_fallback_signal(price_data)

        # 保存信号到历史记录
        signal_data['timestamp'] = price_data['timestamp']
        signal_history.append(signal_data)
        if len(signal_history) > 30:
            signal_history.pop(0)

        # 记录解析后的交易信号
        logger.info("=" * 80)
        logger.info("解析后的交易信号:")
        logger.info(f"信号: {signal_data['signal']}")
        logger.info(f"理由: {signal_data['reason']}")
        logger.info(f"止损: ${signal_data['stop_loss']:,.2f}")
        logger.info(f"止盈: ${signal_data['take_profit']:,.2f}")
        logger.info(f"信心程度: {signal_data['confidence']}")
        logger.info(f"时间戳: {signal_data['timestamp']}")
        logger.info("=" * 80)

        # 信号统计
        signal_count = len([s for s in signal_history if s.get('signal') == signal_data['signal']])
        total_signals = len(signal_history)
        stats_msg = f"信号统计: {signal_data['signal']} (最近{total_signals}次中出现{signal_count}次)"
        print(stats_msg)
        logger.info(stats_msg)

        # 信号连续性检查
        if len(signal_history) >= 3:
            last_three = []
            for s in signal_history[-3:]:
                if isinstance(s, dict) and 'signal' in s:
                    last_three.append(s['signal'])
            if len(last_three) == 3 and len(set(last_three)) == 1:
                warning_msg = f"⚠️ 注意：连续3次{signal_data['signal']}信号"
                print(warning_msg)
                logger.warning(warning_msg)

        return signal_data

    except Exception as e:
        error_msg = f"DeepSeek分析失败: {e}"
        print(error_msg)
        logger.error(error_msg)
        import traceback
        traceback_str = traceback.format_exc()
        print(f"详细错误信息: {traceback_str}")
        logger.error(f"详细错误信息: {traceback_str}")
        return create_fallback_signal(price_data)


def execute_trade(signal_data, price_data):
    """执行交易 - OKX版本（修复保证金检查，支持虚拟仓位）"""
    global position, virtual_position, virtual_balance

    current_position = get_current_position()

    # 记录交易决策
    logger.info("=" * 80)
    logger.info("交易决策分析:")
    logger.info(f"信号: {signal_data['signal']}")
    logger.info(f"信心程度: {signal_data['confidence']}")
    logger.info(f"理由: {signal_data['reason']}")
    logger.info(f"止损: ${signal_data['stop_loss']:,.2f}")
    logger.info(f"止盈: ${signal_data['take_profit']:,.2f}")
    logger.info(f"当前持仓: {current_position}")

    # 🔴 紧急修复：防止频繁反转
    if current_position and signal_data['signal'] != 'HOLD':
        current_side = current_position['side']
        # 修正：正确处理HOLD情况
        if signal_data['signal'] == 'BUY':
            new_side = 'long'
        elif signal_data['signal'] == 'SELL':
            new_side = 'short'
        else:  # HOLD
            new_side = None

        # 如果只是方向反转，需要高信心才执行
        if new_side != current_side:
            if signal_data['confidence'] != 'HIGH':
                msg = f"🔒 非高信心反转信号，保持现有{current_side}仓"
                print(msg)
                logger.info(msg)
                logger.info("=" * 80)
                return

            # 检查最近信号历史，避免频繁反转
            if len(signal_history) >= 2:
                last_signals = [s['signal'] for s in signal_history[-2:]]
                if signal_data['signal'] in last_signals:
                    msg = f"🔒 近期已出现{signal_data['signal']}信号，避免频繁反转"
                    print(msg)
                    logger.info(msg)
                    logger.info("=" * 80)
                    return

    print(f"交易信号: {signal_data['signal']}")
    print(f"信心程度: {signal_data['confidence']}")
    print(f"理由: {signal_data['reason']}")
    print(f"止损: ${signal_data['stop_loss']:,.2f}")
    print(f"止盈: ${signal_data['take_profit']:,.2f}")
    print(f"当前持仓: {current_position}")

    # 风险管理：低信心信号不执行
    if signal_data['confidence'] == 'LOW' and not TRADE_CONFIG['test_mode']:
        msg = "⚠️ 低信心信号，跳过执行"
        print(msg)
        logger.info(msg)
        logger.info("=" * 80)
        return

    # 虚拟仓位模式
    if TRADE_CONFIG['test_mode'] and TRADE_CONFIG.get('virtual_mode', False):
        execute_virtual_trade(signal_data, price_data)
        logger.info("=" * 80)
        return

    if TRADE_CONFIG['test_mode']:
        msg = "测试模式 - 仅模拟交易"
        print(msg)
        logger.info(msg)
        logger.info("=" * 80)
        return

    try:
        # 获取账户余额
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        required_margin = price_data['price'] * TRADE_CONFIG['amount'] / TRADE_CONFIG['leverage']

        if required_margin > usdt_balance * 0.8:  # 使用不超过80%的余额
            print(f"⚠️ 保证金不足，跳过交易。需要: {required_margin:.2f} USDT, 可用: {usdt_balance:.2f} USDT")
            return

        # 执行交易逻辑   tag 是我的经纪商api（不拿白不拿），不会影响大家返佣，介意可以删除
        if signal_data['signal'] == 'BUY':
            if current_position and current_position['side'] == 'short':
                print("平空仓并开多仓...")
                # 平空仓
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    current_position['size'],
                    params={'reduceOnly': True, 'tag': '60bb4a8d3416BCDE'}
                )
                time.sleep(1)
                # 开多仓
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    TRADE_CONFIG['amount'],
                    params={'tag': 'f1ee03b510d5SUDE'}
                )
            elif current_position and current_position['side'] == 'long':
                print("已有多头持仓，保持现状")
            else:
                # 无持仓时开多仓
                print("开多仓...")
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    TRADE_CONFIG['amount'],
                    params={'tag': 'f1ee03b510d5SUDE'}
                )

        elif signal_data['signal'] == 'SELL':
            if current_position and current_position['side'] == 'long':
                print("平多仓并开空仓...")
                # 平多仓
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    current_position['size'],
                    params={'reduceOnly': True, 'tag': 'f1ee03b510d5SUDE'}
                )
                time.sleep(1)
                # 开空仓
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    TRADE_CONFIG['amount'],
                    params={'tag': 'f1ee03b510d5SUDE'}
                )
            elif current_position and current_position['side'] == 'short':
                print("已有空头持仓，保持现状")
            else:
                # 无持仓时开空仓
                print("开空仓...")
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    TRADE_CONFIG['amount'],
                    params={'tag': 'f1ee03b510d5SUDE'}
                )

        print("订单执行成功")
        time.sleep(2)
        position = get_current_position()
        print(f"更新后持仓: {position}")

    except Exception as e:
        print(f"订单执行失败: {e}")
        import traceback
        traceback.print_exc()


def execute_virtual_trade(signal_data, price_data):
    """执行虚拟交易"""
    global virtual_position, virtual_balance
    
    current_price = price_data['price']
    
    msg_list = []
    msg_list.append("\n" + "=" * 60)
    msg_list.append("💰 虚拟仓位交易执行")
    msg_list.append("=" * 60)
    
    # HOLD 信号：保持现有持仓，只更新未实现盈亏
    if signal_data['signal'] == 'HOLD':
        if virtual_position:
            # 计算未实现盈亏
            if virtual_position['side'] == 'long':
                unrealized = (current_price - virtual_position['entry_price']) * virtual_position['size']
            else:
                unrealized = (virtual_position['entry_price'] - current_price) * virtual_position['size']
            
            virtual_position['unrealized_pnl'] = unrealized
            unrealized_percent = (unrealized / (virtual_position['entry_price'] * virtual_position['size'] / TRADE_CONFIG['leverage'])) * 100
            
            msg_list.append(f"保持 {virtual_position['side'].upper()} 仓位 (HOLD):")
            msg_list.append(f"  - 开仓价格: ${virtual_position['entry_price']:,.2f}")
            msg_list.append(f"  - 当前价格: ${current_price:,.2f}")
            msg_list.append(f"  - 仓位大小: {virtual_position['size']:.6f} BTC")
            msg_list.append(f"  - 未实现盈亏: ${unrealized:+,.2f} ({unrealized_percent:+.2f}%)")
            
            logger.info(f"虚拟持仓 - {virtual_position['side'].upper()} | 未实现盈亏: ${unrealized:+,.2f} ({unrealized_percent:+.2f}%)")
        else:
            msg_list.append("保持当前状态 (HOLD) - 无持仓")
            logger.info("虚拟交易 - HOLD (无持仓)")
    
    # BUY/SELL 信号：需要调整仓位
    else:
        # 计算新仓位大小（基于虚拟余额）
        position_value = virtual_balance * 0.95  # 使用95%的余额
        position_size = position_value / current_price * TRADE_CONFIG['leverage']
        
        # 如果有持仓且方向不同，先平仓
        if virtual_position:
            current_side = virtual_position['side']
            new_side = 'long' if signal_data['signal'] == 'BUY' else 'short'
            
            # 只有在方向改变时才平仓
            if current_side != new_side:
                # 计算盈亏
                if virtual_position['side'] == 'long':
                    pnl = (current_price - virtual_position['entry_price']) * virtual_position['size']
                else:  # short
                    pnl = (virtual_position['entry_price'] - current_price) * virtual_position['size']
                
                pnl_percent = (pnl / (virtual_position['entry_price'] * virtual_position['size'] / TRADE_CONFIG['leverage'])) * 100
                
                # 更新余额
                virtual_balance += pnl
                
                msg_list.append(f"平仓 {virtual_position['side'].upper()} 仓位:")
                msg_list.append(f"  - 开仓价格: ${virtual_position['entry_price']:,.2f}")
                msg_list.append(f"  - 平仓价格: ${current_price:,.2f}")
                msg_list.append(f"  - 仓位大小: {virtual_position['size']:.6f} BTC")
                msg_list.append(f"  - 盈亏: ${pnl:+,.2f} ({pnl_percent:+.2f}%)")
                msg_list.append(f"  - 更新后余额: ${virtual_balance:,.2f}")
                
                logger.info(f"虚拟平仓 - {virtual_position['side'].upper()} | 盈亏: ${pnl:+,.2f} ({pnl_percent:+.2f}%) | 余额: ${virtual_balance:,.2f}")
                
                virtual_position = None
                
                # 重新计算仓位大小（余额可能变化）
                position_value = virtual_balance * 0.95
                position_size = position_value / current_price * TRADE_CONFIG['leverage']
            else:
                # 方向相同，保持现有持仓
                msg_list.append(f"已有 {current_side.upper()} 持仓，保持现状")
                logger.info(f"虚拟交易 - 已有{current_side}仓，保持不变")
        
        # 执行新开仓（只有在无持仓或平仓后才开新仓）
        if virtual_position is None:
            if signal_data['signal'] == 'BUY':
                virtual_position = {
                    'side': 'long',
                    'entry_price': current_price,
                    'size': position_size,
                    'leverage': TRADE_CONFIG['leverage'],
                    'symbol': TRADE_CONFIG['symbol'],
                    'unrealized_pnl': 0
                }
                msg_list.append(f"\n开多仓:")
                msg_list.append(f"  - 开仓价格: ${current_price:,.2f}")
                msg_list.append(f"  - 仓位大小: {position_size:.6f} BTC")
                msg_list.append(f"  - 杠杆倍数: {TRADE_CONFIG['leverage']}x")
                msg_list.append(f"  - 保证金: ${position_value / TRADE_CONFIG['leverage']:,.2f}")
                
                logger.info(f"虚拟开多 - 价格: ${current_price:,.2f} | 大小: {position_size:.6f} BTC")
                
            elif signal_data['signal'] == 'SELL':
                virtual_position = {
                    'side': 'short',
                    'entry_price': current_price,
                    'size': position_size,
                    'leverage': TRADE_CONFIG['leverage'],
                    'symbol': TRADE_CONFIG['symbol'],
                    'unrealized_pnl': 0
                }
                msg_list.append(f"\n开空仓:")
                msg_list.append(f"  - 开仓价格: ${current_price:,.2f}")
                msg_list.append(f"  - 仓位大小: {position_size:.6f} BTC")
                msg_list.append(f"  - 杠杆倍数: {TRADE_CONFIG['leverage']}x")
                msg_list.append(f"  - 保证金: ${position_value / TRADE_CONFIG['leverage']:,.2f}")
                
                logger.info(f"虚拟开空 - 价格: ${current_price:,.2f} | 大小: {position_size:.6f} BTC")
    
    # 显示当前统计
    total_pnl = virtual_balance - virtual_initial_balance
    total_pnl_percent = (total_pnl / virtual_initial_balance) * 100
    
    msg_list.append(f"\n📊 账户统计:")
    msg_list.append(f"  - 初始余额: ${virtual_initial_balance:,.2f}")
    msg_list.append(f"  - 当前余额: ${virtual_balance:,.2f}")
    msg_list.append(f"  - 总盈亏: ${total_pnl:+,.2f} ({total_pnl_percent:+.2f}%)")
    
    if virtual_position:
        # 计算当前未实现盈亏
        if virtual_position['side'] == 'long':
            unrealized = (current_price - virtual_position['entry_price']) * virtual_position['size']
        else:
            unrealized = (virtual_position['entry_price'] - current_price) * virtual_position['size']
        
        virtual_position['unrealized_pnl'] = unrealized
        unrealized_percent = (unrealized / (virtual_position['entry_price'] * virtual_position['size'] / TRADE_CONFIG['leverage'])) * 100
        
        msg_list.append(f"  - 持仓盈亏: ${unrealized:+,.2f} ({unrealized_percent:+.2f}%)")
        msg_list.append(f"  - 总资产: ${virtual_balance + unrealized:,.2f}")
    
    msg_list.append("=" * 60)
    
    # 输出所有消息
    for msg in msg_list:
        print(msg)
    
    logger.info(f"虚拟账户 - 余额: ${virtual_balance:,.2f} | 总盈亏: ${total_pnl:+,.2f} ({total_pnl_percent:+.2f}%)")


def analyze_with_deepseek_with_retry(price_data, max_retries=2):
    """带重试的DeepSeek分析"""
    
    # 🔴 修复：添加空值检查
    if not price_data or not isinstance(price_data, dict):
        print("❌ price_data 为空或无效，使用备用信号")
        return create_fallback_signal({'price': 0})
    
    for attempt in range(max_retries):
        try:
            signal_data = analyze_with_deepseek(price_data)
            if signal_data and not signal_data.get('is_fallback', False):
                return signal_data

            print(f"第{attempt + 1}次尝试失败，进行重试...")
            time.sleep(1)

        except Exception as e:
            print(f"第{attempt + 1}次尝试异常: {e}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            if attempt == max_retries - 1:
                return create_fallback_signal(price_data)
            time.sleep(1)

    return create_fallback_signal(price_data)


def wait_for_next_period():
    """等待到下一个15分钟整点"""
    now = datetime.now()
    current_minute = now.minute
    current_second = now.second

    # 计算下一个整点时间（00, 15, 30, 45分钟）
    next_period_minute = ((current_minute // 15) + 1) * 15
    if next_period_minute == 60:
        next_period_minute = 0

    # 计算需要等待的总秒数
    if next_period_minute > current_minute:
        minutes_to_wait = next_period_minute - current_minute
    else:
        minutes_to_wait = 60 - current_minute + next_period_minute

    seconds_to_wait = minutes_to_wait * 60 - current_second

    # 显示友好的等待时间
    display_minutes = minutes_to_wait - 1 if current_second > 0 else minutes_to_wait
    display_seconds = 60 - current_second if current_second > 0 else 0

    if display_minutes > 0:
        print(f"🕒 等待 {display_minutes} 分 {display_seconds} 秒到整点...")
    else:
        print(f"🕒 等待 {display_seconds} 秒到整点...")

    return seconds_to_wait


def wait_with_progress(seconds):
    """带进度显示的等待函数，保持容器活跃"""
    elapsed = 0
    while elapsed < seconds:
        # 每10秒输出一次进度，保持容器活跃
        time.sleep(10)
        elapsed += 10
        remaining = max(0, seconds - elapsed)
        if remaining > 0:
            mins = int(remaining // 60)
            secs = int(remaining % 60)
            print(f"⏱️  已等待 {elapsed//60} 分钟，还需等待 {mins} 分 {secs} 秒...")
            # 每30秒输出一次心跳，确保Railway知道程序还在运行
            if elapsed % 30 == 0:
                print(f"💓 程序运行正常，等待整点执行交易分析...")
    
    if remaining > 0 and remaining <= 10:
        time.sleep(remaining)  # 等待剩余时间


def trading_bot():
    """主交易机器人函数"""
    print("\n" + "=" * 60)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 获取增强版K线数据
    price_data = get_btc_ohlcv_enhanced()
    if not price_data:
        return

    print(f"BTC当前价格: ${price_data['price']:,.2f}")
    print(f"数据周期: {TRADE_CONFIG['timeframe']}")
    print(f"价格变化: {price_data['price_change']:+.2f}%")

    # 2. 使用DeepSeek分析（带重试）
    signal_data = analyze_with_deepseek_with_retry(price_data)

    if signal_data.get('is_fallback', False):
        print("⚠️ 使用备用交易信号")

    # 3. 执行交易
    execute_trade(signal_data, price_data)


def main():
    """主函数"""
    print("BTC/USDT OKX自动交易机器人启动成功！")
    print("融合技术指标策略 + OKX实盘接口")

    if TRADE_CONFIG['test_mode']:
        if TRADE_CONFIG.get('virtual_mode', False):
            print(f"💰 虚拟仓位模式 - 初始金额: ${TRADE_CONFIG.get('virtual_balance', 10000):,.2f}")
            print(f"   使用虚拟资金进行模拟交易，跟踪盈亏")
        else:
            print("当前为模拟模式，不会真实下单")
    else:
        print("实盘交易模式，请谨慎操作！")

    print(f"交易周期: {TRADE_CONFIG['timeframe']}")
    print("已启用完整技术指标分析和持仓跟踪功能")

    # 设置交易所
    if not setup_exchange():
        print("交易所初始化失败，程序退出")
        return

    print(f"执行频率: 每 {TRADE_CONFIG['execution_interval']} 秒执行一次")
    print("=" * 60)
    print("🚀 程序开始运行，开始执行交易分析...")
    print("=" * 60)

    # 循环执行（不使用schedule）
    try:
        while True:
            try:
                trading_bot()
                print(f"✅ 本次分析完成，{TRADE_CONFIG['execution_interval']}秒后执行下次分析...")
            except Exception as e:
                print(f"❌ 交易机器人执行异常: {e}")
                import traceback
                traceback.print_exc()
                print(f"⏳ {TRADE_CONFIG['retry_interval']}秒后重试...")
                time.sleep(TRADE_CONFIG['retry_interval'])  # 出错后等待配置的重试间隔
                continue
            
            # 执行完后等待配置的时间再执行下一次
            print(f"🔄 等待 {TRADE_CONFIG['execution_interval']} 秒...")
            time.sleep(TRADE_CONFIG['execution_interval'])
            
    except KeyboardInterrupt:
        print("\n⚠️ 程序被手动停止")
        if TRADE_CONFIG.get('virtual_mode', False):
            print("\n" + "=" * 60)
            print("💰 虚拟交易最终统计")
            print("=" * 60)
            print(f"初始金额: ${virtual_initial_balance:,.2f}")
            print(f"最终余额: ${virtual_balance:,.2f}")
            total_pnl = virtual_balance - virtual_initial_balance
            total_pnl_percent = (total_pnl / virtual_initial_balance) * 100
            print(f"总盈亏: ${total_pnl:+,.2f} ({total_pnl_percent:+.2f}%)")
            if virtual_position:
                print(f"持仓: {virtual_position['side'].upper()} {virtual_position['size']:.6f} BTC")
            print("=" * 60)
    except Exception as e:
        print(f"\n❌ 程序异常退出: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()