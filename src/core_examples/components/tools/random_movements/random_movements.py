import random

import requests

REQUEST_TIMEOUT_SECONDS = 10


class RandomMovements:
    @staticmethod
    def run(pokemon_name: str) -> list[str]:

        # The url of the api
        url = f'https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}'
            
        # Make the API request
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            raise RuntimeError("Pokemon moves lookup failed") from exc

        # Check if the request was successful
        if response.status_code != 200:
            raise ValueError(f"Error: {pokemon_name} is not a valid pokemon")

        # Parse the response JSON
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("Pokemon moves response is not valid JSON") from exc

        # Extract the list of moves using map and lambda
        moves = list(map(lambda move: move['move']['name'], data['moves']))

        if len(moves) < 4:
            return moves

        # Select 4 random
        selected_moves = random.sample(moves, 4)

        return selected_moves