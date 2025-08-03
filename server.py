```python
  import os
  import redis.asyncio as redis
  from fastapi import FastAPI, WebSocket, WebSocketDisconnect
  from fastapi.responses import HTMLResponse
  import json
  import random
  from collections import defaultdict
  from itertools import combinations
  import uvicorn

  app = FastAPI()
  redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True)

  # Карты и масти
  SUITS = ['♠', '♥', '♦', '♣']
  RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
  DECK = [r + s for r in RANKS for s in SUITS]

  # Game state management
  class GameState:
      def __init__(self):
          self.players = [[] for _ in range(10)]
          self.board = []
          self.folded = []
          self.pot_size = 0
          self.bet_to_call = 0
          self.num_players = 2

      async def load(self):
          state = await redis_client.get("game_state")
          if state:
              data = json.loads(state)
              self.num_players = data.get("num_players", 2)
              self.players = data.get("players", [[] for _ in range(10)])
              self.board = data.get("board", [])
              self.folded = data.get("folded", [])
              self.pot_size = data.get("pot_size", 0)
              self.bet_to_call = data.get("bet_to_call", 0)

      async def save(self):
          await redis_client.set("game_state", json.dumps({
              "num_players": self.num_players,
              "players": self.players,
              "board": self.board,
              "folded": self.folded,
              "pot_size": self.pot_size,
              "bet_to_call": self.bet_to_call
          }))

  class ConnectionManager:
      def __init__(self):
          self.active_connections: list[WebSocket] = []
          self.state = GameState()

      async def connect(self, websocket: WebSocket):
          await websocket.accept()
          self.active_connections.append(websocket)
          await self.state.load()
          await websocket.send_json({"type": "init", "state": vars(self.state)})

      def disconnect(self, websocket: WebSocket):
          self.active_connections.remove(websocket)

      async def broadcast(self, message: dict):
          for connection in self.active_connections:
              await connection.send_json(message)
          await self.state.save()

  manager = ConnectionManager()

  # Оценка комбинации
  def evaluate_hand(hand, board):
      all_cards = hand + board
      if len(all_cards) < 5:
          return 0, []
      
      best_rank = 0
      best_combo = []
      
      for combo in combinations(all_cards, 5):
          rank = evaluate_five_cards(combo)
          if rank > best_rank:
              best_rank = rank
              best_combo = list(combo)
      
      return best_rank, best_combo

  # Оценка комбинации из 5 карт
  def evaluate_five_cards(cards):
      ranks = sorted([RANKS.index(c[:-1]) for c in cards], reverse=True)
      suits = [c[-1] for c in cards]
      is_flush = len(set(suits)) == 1
      is_straight = max(ranks) - min(ranks) == 4 and len(set(ranks)) == 5 or ranks == [12, 3, 2, 1, 0]
      
      rank_counts = defaultdict(int)
      for r in ranks:
          rank_counts[r] += 1
      
      counts = sorted(rank_counts.values(), reverse=True)
      max_count = max(counts)
      
      if is_flush and is_straight and min(ranks) == 8:
          return 10  # Роял-флеш
      elif is_flush and is_straight:
          return 9   # Стрит-флеш
      elif max_count == 4:
          return 8   # Каре
      elif counts == [3, 2]:
          return 7   # Фулл-хаус
      elif is_flush:
          return 6   # Флеш
      elif is_straight:
          return 5   # Стрит
      elif max_count == 3:
          return 4   # Сет
      elif counts == [2, 2]:
          return 3   # Две пары
      elif max_count == 2:
          return 2   # Пара
      return 1       # Старшая карта

  # Монте-Карло симуляция
  def monte_carlo_simulation(active_hands, board, num_players, num_simulations=10000):
      deck = DECK.copy()
      for card in board:
          if card in deck:
              deck.remove(card)
      for hand in active_hands:
          for card in hand:
              if card in deck:
                  deck.remove(card)
      
      wins = [0] * num_players
      for _ in range(num_simulations):
          temp_deck = deck.copy()
          random.shuffle(temp_deck)
          current_board = board.copy()
          
          while len(current_board) < 5:
              current_board.append(temp_deck.pop())
          
          sim_hands = [hand if hand else [temp_deck.pop(), temp_deck.pop()] for hand in active_hands]
          
          hand_ranks = [evaluate_hand(hand, current_board) for hand in sim_hands]
          max_rank = max(rank for rank, _ in hand_ranks)
          winners = [i for i, (rank, _) in enumerate(hand_ranks) if rank == max_rank]
          for winner in winners:
              wins[winner] += 1 / len(winners)
      
      return [win / num_simulations * 100 for win in wins]

  # WebSocket endpoint
  @app.websocket("/ws")
  async def websocket_endpoint(websocket: WebSocket):
      await manager.connect(websocket)
      try:
          while True:
              data = await websocket.receive_json()
              action = data.get("action")
              if action == "update_num_players":
                  manager.state.num_players = data["num_players"]
                  manager.state.players = [[] for _ in range(manager.state.num_players)]
                  manager.state.folded = []
                  await manager.broadcast({"type": "update_num_players", "num_players": manager.state.num_players, "players": manager.state.players, "folded": manager.state.folded})
              elif action == "update_cards":
                  manager.state.players[data["player_id"]] = data["cards"]
                  await manager.broadcast({"type": "update", "state": vars(manager.state)})
              elif action == "update_board":
                  manager.state.board = data["board"]
                  await manager.broadcast({"type": "update_board", "state": vars(manager.state)})
              elif action == "fold":
                  if data["player_id"] not in manager.state.folded:
                      manager.state.folded.append(data["player_id"])
                  await manager.broadcast({"type": "fold", "state": vars(manager.state)})
              elif action == "update_betting":
                  manager.state.pot_size = data["pot_size"]
                  manager.state.bet_to_call = data["bet_to_call"]
                  await manager.broadcast({"type": "update_betting", "state": vars(manager.state)})
              elif action == "reset_game":
                  manager.state.players = [[] for _ in range(manager.state.num_players)]
                  manager.state.board = []
                  manager.state.folded = []
                  manager.state.pot_size = 0
                  manager.state.bet_to_call = 0
                  await manager.broadcast({"type": "reset_game", "state": vars(manager.state)})
              elif action == "calculate":
                  active_hands = [hand if i not in manager.state.folded else [] for i, hand in enumerate(manager.state.players)]
                  active_players = [i for i in range(manager.state.num_players) if i not in manager.state.folded]
                  probabilities = monte_carlo_simulation(
                      [active_hands[i] for i in active_players],
                      manager.state.board,
                      len(active_players)
                  )
                  result = {"probabilities": {i: prob for i, prob in zip(active_players, probabilities)}}
                  if manager.state.pot_size and manager.state.bet_to_call:
                      win_prob = result["probabilities"].get(0, 0) / 100
                      ev = (win_prob * manager.state.pot_size) - ((1 - win_prob) * manager.state.bet_to_call)
                      result["ev"] = ev
                      result["recommendation"] = "Колл" if ev > 0 else "Фолд"
                  await manager.broadcast({"type": "result", "result": result})
      except WebSocketDisconnect:
          manager.disconnect(websocket)
          await manager.broadcast({"type": "update", "state": vars(manager.state)})

  # Serve the frontend
  @app.get("/", response_class=HTMLResponse)
  async def get():
      with open("index.html", encoding="utf-8") as f:
          return f.read()

  # Handle favicon request
  @app.get("/favicon.ico")
  async def favicon():
      return {"message": "No favicon available"}

  if __name__ == "__main__":
      port = int(os.getenv("PORT", 8000))
      uvicorn.run(app, host="0.0.0.0", port=port)
  ```
