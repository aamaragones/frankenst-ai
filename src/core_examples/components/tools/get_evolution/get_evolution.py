import requests

REQUEST_TIMEOUT_SECONDS = 10


class GetEvolution:
    @staticmethod
    def run(pokemon_name: str) -> list[str]:
        species_url = f"https://pokeapi.co/api/v2/pokemon-species/{pokemon_name.lower()}"
        try:
            species_response = requests.get(species_url, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            raise RuntimeError("Pokemon species lookup failed") from exc

        if species_response.status_code != 200:
            raise ValueError(f"Error: {pokemon_name} is not a valid pokemon")
        
        try:
            species_data = species_response.json()
        except ValueError as exc:
            raise RuntimeError("Pokemon species response is not valid JSON") from exc

        # Step 2: Extract evolution chain URL from species data
        evolution_chain_url = species_data['evolution_chain']['url']

        # Step 3: Get the evolution chain data
        try:
            evolution_response = requests.get(evolution_chain_url, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            raise RuntimeError("Pokemon evolution lookup failed") from exc

        try:
            evolution_data = evolution_response.json()
        except ValueError as exc:
            raise RuntimeError("Pokemon evolution response is not valid JSON") from exc

        # Step 4: Traverse the evolution chain and get the names of evolutions
        evolutions = []
        current_evolution = evolution_data['chain']
        
        while current_evolution:
            evolutions.append(current_evolution['species']['name'])
            if len(current_evolution['evolves_to']) > 0:
                current_evolution = current_evolution['evolves_to'][0]
            else:
                break
        
        return evolutions