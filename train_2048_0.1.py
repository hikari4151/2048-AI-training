import sys
import time
import random
import copy
import os
import threading
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QPushButton
)

# ======================== 游戏环境 ========================
class Game2048:
    def __init__(self, render=False):
        self.grid = [[0]*4 for _ in range(4)]
        self.score = 0
        self.render = render
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((400, 500), pygame.RESIZABLE)
            pygame.display.set_caption("布里茨1号展示中")
            self.clock = pygame.time.Clock()
            self.font_size = 36
            self.font = pygame.font.SysFont("Arial", self.font_size, bold=True)
        self.add_random_tile()
        self.add_random_tile()

    def add_random_tile(self):
        empty = [(i, j) for i in range(4) for j in range(4) if self.grid[i][j] == 0]
        if empty:
            i, j = random.choice(empty)
            self.grid[i][j] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        new_row = [v for v in row if v != 0]
        new_row += [0]*(4 - len(new_row))
        return new_row

    def merge(self, row):
        reward = 0
        for i in range(3):
            if row[i] != 0 and row[i] == row[i+1]:
                row[i] *= 2
                row[i+1] = 0
                reward += row[i]
        return row, reward

    def move_left(self):
        new_grid = []
        total_reward = 0
        for row in self.grid:
            comp = self.compress(row)
            merged, r = self.merge(comp)
            new_grid.append(self.compress(merged))
            total_reward += r
        self.grid = new_grid
        self.score += total_reward
        return total_reward

    def reverse(self):
        self.grid = [row[::-1] for row in self.grid]

    def transpose(self):
        self.grid = [list(row) for row in zip(*self.grid)]

    def move(self, action):
        old_grid = copy.deepcopy(self.grid)
        reward = 0
        if action == 0:
            reward = self.move_left()
        elif action == 1:
            self.transpose()
            reward = self.move_left()
            self.transpose()
        elif action == 2:
            self.reverse()
            reward = self.move_left()
            self.reverse()
        elif action == 3:
            self.transpose()
            self.reverse()
            reward = self.move_left()
            self.reverse()
            self.transpose()
        moved = (self.grid != old_grid)
        if moved:
            self.add_random_tile()
        return reward if moved else 0, moved

    def can_move(self):
        for i in range(4):
            for j in range(4):
                if self.grid[i][j] == 0:
                    return True
                if j < 3 and self.grid[i][j] == self.grid[i][j+1]:
                    return True
                if i < 3 and self.grid[i][j] == self.grid[i+1][j]:
                    return True
        return False

    def step(self, action, thinking_time=0.0):
        reward, moved = self.move(action)
        done = not self.can_move()

        # ---------- 奖励塑形 ----------
        shaped_reward = reward
        empty_count = sum(row.count(0) for row in self.grid)
        shaped_reward += 0.5 * empty_count
        smooth_penalty = 0
        for i in range(4):
            for j in range(4):
                if self.grid[i][j]:
                    if j < 3 and self.grid[i][j+1]:
                        smooth_penalty += abs(self.grid[i][j] - self.grid[i][j+1])
                    if i < 3 and self.grid[i+1][j]:
                        smooth_penalty += abs(self.grid[i][j] - self.grid[i+1][j])
        shaped_reward -= 0.1 * smooth_penalty
        shaped_reward += 1.0
        if not moved:
            shaped_reward -= 2.0
        if done:
            shaped_reward -= 10.0

        # ---------- 新奖惩：思考超时惩罚 ----------
        if thinking_time > 5.0:
            shaped_reward -= 1.0

        return self.grid, shaped_reward, done

    def render_game(self):
        if not self.render:
            return

        # 获取当前窗口的实际宽高（拉伸后会变）
        win_w, win_h = self.screen.get_size()
        # 棋盘区域占窗口的大部分，底部留一段显示分数
        board_size = min(win_w, win_h - 50)  # 底部留 50px 给分数
        cell_size = board_size // 4
        margin_x = (win_w - cell_size * 4) // 2
        margin_y = (win_h - 50 - cell_size * 4) // 2  # 垂直居中（为分数栏留空间）

        self.screen.fill((250, 248, 239))

        # 动态字号（按格子大小的一定比例）
        dynamic_font_size = max(10, cell_size // 3)
        font = pygame.font.SysFont("Arial", dynamic_font_size, bold=True)

        for i in range(4):
            for j in range(4):
                val = self.grid[i][j]
                # 颜色映射（同原来）
                color = (204, 192, 179) if val == 0 else \
                    (238, 228, 218) if val == 2 else (237, 224, 200) if val == 4 else \
                        (242, 177, 121) if val == 8 else (245, 149, 99) if val == 16 else \
                            (246, 124, 95) if val == 32 else (246, 94, 59) if val == 64 else \
                                (237, 207, 114) if val == 128 else (237, 204, 97) if val == 256 else \
                                    (237, 200, 80) if val == 512 else (237, 197, 63) if val == 1024 else (237, 194, 46)

                rect = pygame.Rect(
                    margin_x + j * cell_size,
                    margin_y + i * cell_size,
                    cell_size,
                    cell_size
                )
                pygame.draw.rect(self.screen, color, rect, border_radius=8)

                if val != 0:
                    text = font.render(str(val), True, (255, 255, 255) if val > 4 else (119, 110, 101))
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)

        # 显示分数（用较小字号，靠下居中）
        score_font = pygame.font.SysFont("Arial", max(12, cell_size // 4), bold=True)
        score_text = score_font.render(f"Score: {self.score}", True, (0, 0, 0))
        score_rect = score_text.get_rect(center=(win_w // 2, win_h - 25))
        self.screen.blit(score_text, score_rect)

        pygame.display.flip()
        self.clock.tick(0)  # 不锁帧，让拉伸时足够流畅

# ======================== 状态预处理 ========================
def state_to_tensor(grid):
    arr = np.array(grid, dtype=np.float32)
    log_grid = np.zeros_like(arr)
    nonzero = arr > 0
    log_grid[nonzero] = np.log2(arr[nonzero])
    log_grid /= 16.0
    return torch.tensor(log_grid.flatten(), dtype=torch.float32)

# ======================== DQN 网络 ========================
class DQN(nn.Module):
    def __init__(self, state_dim=16, action_dim=4, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ======================== DQN 智能体 ========================
class DQNAgent:
    def __init__(self, lr=1e-3, gamma=0.99, batch_size=64, buffer_capacity=20000,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.99995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=buffer_capacity)
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.train_steps = 0

    def select_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, 3)
        with torch.no_grad():
            q_vals = self.policy_net(state.unsqueeze(0).to(self.device))
            return q_vals.argmax().item()

    def store_transition(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        self.train_steps += 1
        batch = random.sample(self.memory, self.batch_size)
        s_batch = torch.stack([b[0] for b in batch]).to(self.device)
        a_batch = torch.tensor([b[1] for b in batch], dtype=torch.long).unsqueeze(1).to(self.device)
        r_batch = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1).to(self.device)
        s_next_batch = torch.stack([b[3] for b in batch]).to(self.device)
        done_batch = torch.tensor([b[4] for b in batch], dtype=torch.float32).unsqueeze(1).to(self.device)

        q_eval = self.policy_net(s_batch).gather(1, a_batch)
        with torch.no_grad():
            q_next = self.target_net(s_next_batch).max(1)[0].unsqueeze(1)
            q_target = r_batch + self.gamma * q_next * (1 - done_batch)

        loss = self.loss_fn(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_checkpoint(self, path, episode, total_elapsed, high_score):
        torch.save({
            'episode': episode,
            'total_elapsed': total_elapsed,
            'high_score': high_score,
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_steps = checkpoint.get('train_steps', 0)
        return (checkpoint['episode'], checkpoint['total_elapsed'],
                checkpoint['high_score'])

# ======================== 训练线程 ========================
class TrainThread(QThread):
    update_signal = pyqtSignal(int, float, int, float)

    def __init__(self, episodes=3000, render=False):
        super().__init__()
        self.episodes = episodes
        self.render = render
        self.agent = None
        self.running = True
        self.start_ep = 1
        self.total_elapsed = 0.0
        self.high_score = 0
        self.checkpoint_path = "dqn_checkpoint.pth"

        self.paused = threading.Event()
        self.paused.set()
        self.pause_start_time = None
        self.total_pause = 0.0

    def pause(self):
        if self.paused.is_set():
            self.paused.clear()
            self.pause_start_time = time.time()

    def resume(self):
        if not self.paused.is_set():
            self.paused.set()
            if self.pause_start_time is not None:
                self.total_pause += time.time() - self.pause_start_time
                self.pause_start_time = None

    def is_paused(self):
        return not self.paused.is_set()

    def run(self):
        try:
            self._run_impl()
        except Exception as e:
            with open("train_error.log", "w", encoding="utf-8") as f:
                import traceback
                traceback.print_exc(file=f)
            raise

    def _run_impl(self):
        self.agent = DQNAgent()
        if os.path.exists(self.checkpoint_path):
            print("发现已有存档，正在继承训练进度...")
            try:
                self.start_ep, self.total_elapsed, self.high_score = \
                    self.agent.load_checkpoint(self.checkpoint_path)
                self.start_ep += 1
                print(f"继承完成：从 Episode {self.start_ep} 继续训练，已用时 {self.total_elapsed:.0f} 秒，最高分 {self.high_score}")
            except Exception as e:
                print(f"加载存档失败: {e}，将从头开始训练")

        env = Game2048(render=self.render)
        score_buffer = deque(maxlen=100)
        start_time = time.time()
        sync_interval = 100
        save_interval = 50
        last_update_time = time.time()

        for ep in range(self.start_ep, self.episodes + 1):
            while not self.paused.is_set():
                if not self.running:
                    return
                time.sleep(0.1)
            if not self.running:
                break

            state = state_to_tensor(env.grid)
            done = False
            while not done:
                if self.render:
                    env.render_game()

                # 记录思考开始时间
                t0 = time.time()
                action = self.agent.select_action(state)
                thinking_time = time.time() - t0

                _, reward, done = env.step(action, thinking_time=thinking_time)
                next_state = state_to_tensor(env.grid)
                self.agent.store_transition(state, action, reward, next_state, done)
                self.agent.update()
                state = next_state

            score_buffer.append(env.score)
            if env.score > self.high_score:
                self.high_score = env.score

            if ep % sync_interval == 0:
                self.agent.sync_target()

            now = time.time()
            actual_train_time = self.total_elapsed + (now - start_time - self.total_pause)
            if now - last_update_time >= 1.0:
                avg_score = np.mean(score_buffer) if score_buffer else 0.0
                self.update_signal.emit(ep, actual_train_time, self.high_score, avg_score)
                last_update_time = now

            if ep % save_interval == 0 or ep == self.episodes:
                self.agent.save_checkpoint(self.checkpoint_path, ep, actual_train_time, self.high_score)

            env = Game2048(render=self.render)

        final_train_time = self.total_elapsed + (time.time() - start_time - self.total_pause)
        self.agent.save_checkpoint(self.checkpoint_path, self.episodes, final_train_time, self.high_score)
        print("训练完成，模型已保存")
        if self.render:
            pygame.quit()

    def stop(self):
        self.running = False
        self.resume()

# ======================== 展示线程（查看 AI 水平） ========================
class DemoThread(QThread):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def run(self):
        env = Game2048(render=True)
        state = state_to_tensor(env.grid)
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            action = self.agent.select_action(state, eval_mode=True)
            _, _, done = env.step(action, thinking_time=0.0)  # 演示不惩罚
            state = state_to_tensor(env.grid)
            env.render_game()
            time.sleep(0.05)
        time.sleep(2)
        pygame.quit()

# ======================== 悬浮窗 ========================
class FloatingWindow(QWidget):
    def __init__(self, train_thread=None):
        super().__init__()
        self.train_thread = train_thread
        self.demo_thread = None
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(300, 280)

        label_style = """
            QLabel {
                color: white;
                font-size: 14px;
                background-color: rgba(40, 40, 40, 180);
                border-radius: 12px;
                padding: 10px;
            }
        """
        self.info_label = QLabel("训练信息", self)
        self.info_label.setStyleSheet(label_style)
        self.info_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        btn_style = """
            QPushButton {
                color: white;
                background-color: rgba(60, 60, 200, 200);
                border-radius: 8px;
                padding: 8px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: rgba(80, 80, 220, 220);
            }
        """
        self.show_btn = QPushButton("查看 AI 水平", self)
        self.show_btn.setStyleSheet(btn_style)
        self.show_btn.clicked.connect(self.show_ai_demo)

        pause_style = """
            QPushButton {
                color: white;
                background-color: rgba(220, 100, 50, 200);
                border-radius: 8px;
                padding: 8px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: rgba(240, 120, 70, 220);
            }
        """
        self.pause_btn = QPushButton("暂停训练", self)
        self.pause_btn.setStyleSheet(pause_style)
        self.pause_btn.clicked.connect(self.do_pause)

        resume_style = """
            QPushButton {
                color: white;
                background-color: rgba(50, 180, 100, 200);
                border-radius: 8px;
                padding: 8px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: rgba(70, 200, 120, 220);
            }
        """
        self.resume_btn = QPushButton("继续训练", self)
        self.resume_btn.setStyleSheet(resume_style)
        self.resume_btn.clicked.connect(self.do_resume)
        self.resume_btn.hide()

        layout = QVBoxLayout(self)
        layout.addWidget(self.info_label)
        layout.addWidget(self.show_btn)
        layout.addWidget(self.pause_btn)
        layout.addWidget(self.resume_btn)
        self.setLayout(layout)
        self.old_pos = None

    def do_pause(self):
        if self.train_thread:
            self.train_thread.pause()
            self.pause_btn.hide()
            self.resume_btn.show()

    def do_resume(self):
        if self.train_thread:
            self.train_thread.resume()
            self.resume_btn.hide()
            self.pause_btn.show()

    def update_info(self, episode, elapsed_time, high_score, avg_score=None):
        try:
            hours, rem = divmod(int(elapsed_time), 3600)
            mins, secs = divmod(rem, 60)
            time_str = f"{hours:02d}:{mins:02d}:{secs:02d}"
            text = f"Episode: {episode}\n"
            text += f"训练时间: {time_str}\n"
            text += f"最高分数: {high_score}\n"
            if avg_score is not None:
                text += f"平均分数: {avg_score:.1f}\n"
            else:
                text += "\n"
            if self.train_thread and self.train_thread.is_paused():
                text += "⏸️ 已暂停"
            self.info_label.setText(text)
        except Exception as e:
            print("update_info 异常:", e)

    def show_ai_demo(self):
        if self.train_thread is None or self.train_thread.agent is None:
            return
        self.demo_thread = DemoThread(self.train_thread.agent)
        self.demo_thread.start()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.old_pos = event.globalPos()

    def mouseMoveEvent(self, event):
        if self.old_pos is not None:
            delta = event.globalPos() - self.old_pos
            self.move(self.pos() + delta)
            self.old_pos = event.globalPos()

    def mouseReleaseEvent(self, event):
        self.old_pos = None

# ======================== 主程序 ========================
if __name__ == "__main__":
    import sys
    print(f"训练设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  显卡: {torch.cuda.get_device_name(0)}")
    app = QApplication(sys.argv)

    train_thread = TrainThread(episodes=5000, render=False)
    window = FloatingWindow(train_thread)
    window.move(100, 100)
    window.show()

    train_thread.update_signal.connect(window.update_info)
    train_thread.start()

    app.aboutToQuit.connect(train_thread.stop)
    sys.exit(app.exec_())