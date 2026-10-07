import os
import time
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from google.cloud import storage

# --- Configuration & Frictions ---
BUCKET_NAME = os.getenv("BUCKET_NAME")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

COMMISSION_FEE = 0.0002   # 0.02% per trade
SLIPPAGE_BPS = 0.00005     # 0.005% slippage
GAMMA = 0.95
LR = 0.001
BATCH_SIZE = 32

def send_telegram_alert(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"📡 Webhook Log (Not Sent):\n{message}")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload, timeout=5)
        print(f"📡 Telegram Response Status: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Telegram network connection failed: {e}")

def append_and_upload_csv(tick, equity, agent, local_filename="equity_history.csv"):
    row_data = {"Tick": tick, "Total_Equity": equity}
    for idx, ticker in enumerate(agent.tickers):
        count = agent.transaction_counts.get(ticker, 0)
        pnl = agent.realized_pnl.get(ticker, 0.0)
        avg_return = pnl / count if count > 0 else 0.0
        row_data[f"{ticker}_TxCount"] = count
        row_data[f"{ticker}_CumPnL"] = pnl
        row_data[f"{ticker}_AvgReturnPerTx"] = avg_return

    new_row = pd.DataFrame([row_data])
    if BUCKET_NAME:
        try:
            client = storage.Client()
            bucket = client.bucket(BUCKET_NAME)
            blob = bucket.blob(local_filename)
            if blob.exists():
                blob.download_to_filename(local_filename)
                existing_df = pd.read_csv(local_filename)
                updated_df = pd.concat([existing_df, new_row], ignore_index=True)
            else:
                updated_df = new_row
            updated_df.to_csv(local_filename, index=False)
            blob.upload_from_filename(local_filename)
        except Exception as e:
            print(f"⚠️ GCS CSV Append Failed: {e}")
    else:
        if os.path.exists(local_filename):
            existing_df = pd.read_csv(local_filename)
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
        else:
            updated_df = new_row
        updated_df.to_csv(local_filename, index=False)

def save_to_gcs(local_path, cloud_name):
    if not BUCKET_NAME: return
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(cloud_name)
        blob.upload_from_filename(local_path)
    except Exception as e: print(f"⚠️ GCS Upload Failed: {e}")

def load_from_gcs(local_path, cloud_name):
    if not BUCKET_NAME: return False
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(cloud_name)
        if blob.exists():
            blob.download_to_filename(local_path)
            return True
    except Exception as e: print(f"⚠️ GCS Download Failed: {e}")
    return False

#Neural Network Architecture & The Deep Learning Agent
class PrioritizedReplayBuffer:
    def __init__(self, capacity=1000, alpha=0.6):
        self.capacity, self.alpha, self.buffer, self.priorities = capacity, alpha, [], []
    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0); self.priorities.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
    def sample(self, batch_size):
        scaled = np.array(self.priorities) ** self.alpha
        probs = scaled / np.sum(scaled)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        return [self.buffer[idx] for idx in indices], indices
    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors): self.priorities[idx] = abs(error) + 1e-6

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, x): return self.net(x)

class MultiAssetAgent:
    def __init__(self, tickers, state_dim=4, action_dim=3):
        self.tickers, self.action_dim = tickers, action_dim
        self.memory = PrioritizedReplayBuffer()
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        
        # Core Position Parameters
        self.positions = {t: 0 for t in tickers}
        self.last_states = {t: None for t in tickers}
        self.last_actions = {t: 0 for t in tickers}
        self.last_prices = {t: None for t in tickers}
        
        # Advanced Transactional Metrics Dictionaries
        self.entry_prices = {t: 0.0 for t in tickers}
        self.transaction_counts = {t: 0 for t in tickers}
        self.realized_pnl = {t: 0.0 for t in tickers}

        #Win/Loss & Profit Factor Component Trackers
        self.winning_trades_count = {t: 0 for t in tickers}
        self.losing_trades_count = {t: 0 for t in tickers}
        self.gross_profits = {t: 0.0 for t in tickers}
        self.gross_losses = {t: 0.0 for t in tickers}

    def choose_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad(): return torch.argmax(self.model(state_t)).item()

    def train_step(self):
        if len(self.memory.buffer) < BATCH_SIZE: return
        samples, indices = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*samples)
        states_t, actions_t = torch.FloatTensor(np.array(states)), torch.LongTensor(actions).unsqueeze(1)
        rewards_t, next_states_t, dones_t = torch.FloatTensor(rewards).unsqueeze(1), torch.FloatTensor(np.array(next_states)), torch.FloatTensor(dones).unsqueeze(1)
        
        current_q = self.model(states_t).gather(1, actions_t)
        with torch.no_grad():
            # Fixed index syntax tracker layout bug [100% resolved]
            max_next_q = self.model(next_states_t).max(1)[0].unsqueeze(1)
            target_q = rewards_t + (GAMMA * max_next_q * (1 - dones_t))
            
        td_errors = (target_q - current_q).squeeze(1).detach().numpy()
        self.memory.update_priorities(indices, td_errors)
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

#Live Market Loops, Advanced Accounting, and Runtime Trigger
def run_cloud_trading(tickers=['BTC-USD', 'ETH-USD', 'SOL-USD'], interval_seconds=60):
    send_telegram_alert("🟢 *DQN Multi-Asset Trading Bot Online! Fixed Execution Loop Activated.*")
    agent = MultiAssetAgent(tickers=tickers, state_dim=4)
    
    if load_from_gcs('dqn_model.pth', 'dqn_model.pth'):
        agent.model.load_state_dict(torch.load('dqn_model.pth'))
        print("🧠 Weights restored from GCS successfully.")
        
    total_portfolio_equity = 30000.0
    tick_count = 0
    
    if load_from_gcs('equity_history.csv', 'equity_history.csv'):
        try:
            history_df = pd.read_csv('equity_history.csv')
            if not history_df.empty:
                tick_count = int(history_df['Tick'].iloc[-1])
                total_portfolio_equity = float(history_df['Total_Equity'].iloc[-1])
                for ticker in tickers:
                    if f"{ticker}_TxCount" in history_df.columns:
                        agent.transaction_counts[ticker] = int(history_df[f"{ticker}_TxCount"].iloc[-1])
                        agent.realized_pnl[ticker] = float(history_df[f"{ticker}_CumPnL"].iloc[-1])
                print(f"📊 State Restored! Resuming safely from Tick {tick_count}")
        except Exception as e: print(f"⚠️ Resumption parsing failure ({e}).")

    while True:
        tick_count += 1
        current_tick_returns = []
        
        for idx, ticker_name in enumerate(tickers):
            try:
                df = yf.Ticker(ticker_name).history(period="1d", interval="1m")
            except Exception as e:
                print(f"⚠️ Fetch Failure for {ticker_name}: {e}. Skipping pass.")
                continue
                
            df.dropna(subset=['Close'], inplace=True) 
            if df.empty or len(df) < 2: 
                print(f"⚠️ Market window length insufficient for {ticker_name} (Got {len(df)} rows). Skipping ticker pass.")
                continue
                
            current_price = df['Close'].iloc[-1]
            window_size = min(15, len(df))
            prices_window = df['Close'].iloc[-window_size:].values
            
            diffs = np.diff(prices_window) if len(prices_window) > 1 else np.array([0.0])
            gains = np.sum(diffs[diffs > 0]) / float(window_size)
            losses = np.sum(-diffs[diffs < 0]) / float(window_size)
            rsi = (gains / (gains + losses + 1e-9))
            
            mom_depth = min(5, len(prices_window))
            momentum = (prices_window[-1] - prices_window[-mom_depth]) / prices_window[-mom_depth] if len(prices_window) >= mom_depth else 0.0
            
            current_state = np.array([rsi, momentum, float(agent.positions[ticker_name]), float(idx)])
            
            if agent.last_states[ticker_name] is not None and agent.last_prices[ticker_name] is not None:
                price_return = (current_price - agent.last_prices[ticker_name]) / agent.last_prices[ticker_name]
                friction = (COMMISSION_FEE + SLIPPAGE_BPS) if agent.last_actions[ticker_name] != agent.positions[ticker_name] else 0.0
                reward = (price_return - friction) if agent.last_actions[ticker_name] == 1 else (-price_return - friction) if agent.last_actions[ticker_name] == 2 else 0.0
                current_tick_returns.append(reward)
                agent.memory.push(agent.last_states[ticker_name], agent.last_actions[ticker_name], reward, current_state, False)
            
            action = agent.choose_action(current_state)
            
            # --- EVALUATE POSITION TRANSITIONS & COMPOSE MATHEMATICAL ALERTS ---
            if action != agent.positions[ticker_name]:
                pos_names = {0: "CASH ⚪", 1: "LONG 🟢", 2: "SHORT 🔴"}
                old_pos = agent.positions[ticker_name]
                alert_text = f"⚡ *TRADE ALERT* ⚡\nAsset: `{ticker_name}`\nPosition Shift: {pos_names[old_pos]} ➡️ {pos_names[action]}\n"
                
                """ if old_pos in [1, 2]:
                    entry_p = agent.entry_prices[ticker_name] if agent.entry_prices[ticker_name] > 0.0 else current_price
                    price_diff = (current_price - entry_p) if old_pos == 1 else (entry_p - current_price) """
                    # Case A: Opening a Position
                if old_pos == 0 and (action == 1 or action == 2):
                    agent.entry_prices[ticker_name] = current_price
                    agent.transaction_counts[ticker_name] += 1
                    alert_text += f"📥 *Position Opened* | Entry Price: \${current_price:,.2f}"
                
                    # Case B: Closing or Flipping an Active Position
                elif old_pos in [1, 2]:
                        entry_p = agent.entry_prices[ticker_name]
                        if entry_p == 0.0: entry_p = current_price
                        
                        if old_pos == 1:
                            price_diff = current_price - entry_p
                        else:
                            price_diff = entry_p - current_price
                        total_fees = (entry_p * COMMISSION_FEE) + (current_price * COMMISSION_FEE) + (current_price * SLIPPAGE_BPS)
                        trade_pnl = price_diff - total_fees
                        
                        agent.realized_pnl[ticker_name] += trade_pnl
                        agent.transaction_counts[ticker_name] += 1
                        
                        if trade_pnl > 0:
                            agent.winning_trades_count[ticker_name] += 1
                            agent.gross_profits[ticker_name] += trade_pnl
                        else:
                            agent.losing_trades_count[ticker_name] += 1
                            agent.gross_losses[ticker_name] += abs(trade_pnl)
                            
                        wins, losses = agent.winning_trades_count[ticker_name], agent.losing_trades_count[ticker_name]
                        win_loss_ratio = wins / losses if losses > 0 else float(wins)
                        profit_factor = agent.gross_profits[ticker_name] / agent.gross_losses[ticker_name] if agent.gross_losses[ticker_name] > 0 else 1.0
                        
                        dir_sym = "" if old_pos == 1 else "-"
                        alert_text += (
                            f"📊 *Mathematical P&L Breakdown:*\nFormula: {dir_sym}(\${current_price:,.2f} - \${entry_p:,.2f}) - Fees\n"
                            f"Net Trade Output: *\${trade_pnl:+.4f}*\n\n📈 *Asset Performance Metrics:*\n"
                            f"Asset Lifetime P&L: `\${agent.realized_pnl[ticker_name]:,.2f}`\n"
                            f"Record: `{wins}W / {losses}L` | Ratio: `{win_loss_ratio:.2f}`\nProfit Factor: *{profit_factor:.2f}*\n"
                        )
                    
                        if action in [1, 2]:
                            agent.entry_prices[ticker_name] = current_price
                            if old_pos == 0: agent.transaction_counts[ticker_name] += 1
                            alert_text += f"📥 *Position Opened* | Entry Price: \${current_price:,.2f}"
                        else:
                            agent.entry_prices[ticker_name] = 0.0
                
                send_telegram_alert(alert_text)
                
            agent.last_states[ticker_name], agent.last_actions[ticker_name], agent.last_prices[ticker_name], agent.positions[ticker_name] = current_state, action, current_price, action

        if current_tick_returns:
            total_portfolio_equity *= (1.0 + np.mean(current_tick_returns))
            agent.train_step()

        print(f"⏱️ Tick {tick_count} | Equity: \${total_portfolio_equity:,.2f} | Positions: {agent.positions}")
        
        if tick_count % 10 == 0:
            torch.save(agent.model.state_dict(), 'dqn_model.pth')
            save_to_gcs('dqn_model.pth', 'dqn_model.pth')
            append_and_upload_csv(tick_count, total_portfolio_equity, agent, "equity_history.csv")
            print("💾 Saved state files safely to GCS bucket.")

        if tick_count % 60 == 0:
            heartbeat_text = (
                f"🤖 *DQN BOT HEARTBEAT REPORT* 🤖\n"
                f"Elapsed Ticks: `{tick_count}`\n"
                f"Current Master Equity: *\${total_portfolio_equity:,.2f}*\n"
                f"Active Positions: `{agent.positions}`\n"
                f"📊 System Status: *Healthy and Monitoring Markets...*"
            )
            send_telegram_alert(heartbeat_text)    
        time.sleep(interval_seconds)

if __name__ == "__main__":
    run_cloud_trading()

