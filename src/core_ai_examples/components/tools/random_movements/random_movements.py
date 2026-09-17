import random

import requests

REQUEST_TIMEOUT_SECONDS = 10


class RandomMovements:
    @staticmethod
    def run(pokemon_name: str) -> list[str]:
        url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            raise RuntimeError("Pokemon moves lookup failed") from exc
        if response.status_code != 200:
            raise ValueError(f"Error: {pokemon_name} is not a valid pokemon")
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("Pokemon moves response is not valid JSON") from exc

        moves = [move["move"]["name"] for move in data["moves"]]
        if len(moves) < 4:
            return moves
        return random.sample(moves, 4)
