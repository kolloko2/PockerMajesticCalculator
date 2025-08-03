from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import json
import random
from collections import defaultdict
from itertools import combinations
import uvicorn

app = FastAPI()

# Карты и масти
SUITS = ['♠', '♥', '♦', '♣']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
DECK = [r + s for r in RANKS for s in SUITS]

# Game state management
game_state = {"players": [[] for _ in range(10)], "board": [], "folded": [], "pot_size": 0, "bet_to_call": 0, "num_players": 2}
connections = []  # List of active WebSocket connections

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
    await websocket.accept()
    connections.append(websocket)
    await broadcast({"type": "init", "state": game_state})
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message["action"] == "update_num_players":
                game_state["num_players"] = message["num_players"]
                game_state["players"] = [[] for _ in range(game_state["num_players"])]
                game_state["folded"] = []
                await broadcast({"type": "update_num_players", "num_players": game_state["num_players"], "players": game_state["players"], "folded": game_state["folded"]})
            elif message["action"] == "update_cards":
                game_state["players"][message["player_id"]] = message["cards"]
                await broadcast({"type": "update", "state": game_state})
            elif message["action"] == "update_board":
                game_state["board"] = message["board"]
                await broadcast({"type": "update_board", "state": game_state})
            elif message["action"] == "fold":
                if message["player_id"] not in game_state["folded"]:
                    game_state["folded"].append(message["player_id"])
                await broadcast({"type": "fold", "state": game_state})
            elif message["action"] == "update_betting":
                game_state["pot_size"] = message["pot_size"]
                game_state["bet_to_call"] = message["bet_to_call"]
                await broadcast({"type": "update_betting", "state": game_state})
            elif message["action"] == "reset_game":
                game_state["players"] = [[] for _ in range(game_state["num_players"])]
                game_state["board"] = []
                game_state["folded"] = []
                game_state["pot_size"] = 0
                game_state["bet_to_call"] = 0
                await broadcast({"type": "reset_game", "state": game_state})
            elif message["action"] == "calculate":
                active_hands = [hand if i not in game_state["folded"] else [] for i, hand in enumerate(game_state["players"])]
                active_players = [i for i in range(game_state["num_players"]) if i not in game_state["folded"]]
                probabilities = monte_carlo_simulation(
                    [active_hands[i] for i in active_players],
                    game_state["board"],
                    len(active_players)
                )
                result = {"probabilities": {i: prob for i, prob in zip(active_players, probabilities)}}
                if game_state["pot_size"] and game_state["bet_to_call"]:
                    win_prob = result["probabilities"].get(0, 0) / 100
                    ev = (win_prob * game_state["pot_size"]) - ((1 - win_prob) * game_state["bet_to_call"])
                    result["ev"] = ev
                    result["recommendation"] = "Колл" if ev > 0 else "Фолд"
                await broadcast({"type": "result", "result": result})
    except WebSocketDisconnect:
        connections.remove(websocket)

async def broadcast(message):
    for connection in connections:
        await connection.send_text(json.dumps(message))

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
