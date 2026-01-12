import os
import asyncio
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types
from pydantic import BaseModel
import colorama
from typing import Dict, List

from fsm_llm import LLMStateMachine
from fsm_llm.state_models import FSMRun

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

class GeminiOpenAIWrapper:
    def __init__(self, api_key: str, model_id: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.beta = self
        self.chat = self
        self.completions = self

    async def parse(self, messages, response_format=None, **kwargs):
        prompt = messages[-1]["content"]
        config = types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json" if response_format else "text/plain",
            response_schema=response_format if response_format else None,
        )
        try:
            response = await asyncio.to_thread(
                lambda: self.client.models.generate_content(model=self.model_id, contents=prompt, config=config)
            )
            text_out = response.text
        except Exception as e:
            print(f"{colorama.Fore.RED}[API ERROR]: {e}{colorama.Fore.RESET}")
            text_out = "{}" if response_format else "..."

        class ParsedMessage:
            def __init__(self, text, model):
                self.content = text
                self.parsed = model.model_validate_json(text) if model and text != "{}" else None
        class MockChoice:
            def __init__(self, text, model):
                self.message = ParsedMessage(text, model)
        class MockResponse:
            def __init__(self, text, model):
                self.choices = [MockChoice(text, model)]
        return MockResponse(text_out, response_format)

class Item(BaseModel):
    id: str
    name: str
    description: str
    effect: str

class ItemAmount(BaseModel):
    id: str
    amount: int

class Recipe(BaseModel):
    result_id: str
    ingredients: Dict[str, int]

class ActionResponseModel(BaseModel):
    item_id: str | None = None
    amount: int = 1

fsm = LLMStateMachine(initial_state="GREETING", end_state="END")

# DATABASE OGGETTI
fsm.set_context_data("items", {
    "healing-potion": Item(id="healing-potion", name="Healing Potion", description="A bubbling red liquid.", effect="Restores 50 HP immediately."),
    "defense-potion": Item(id="defense-potion", name="Defense Potion", description="Thick and smells like earth.", effect="+20 Physical Resistance for 5 min."),
    "acorn": Item(id="acorn", name="Acorn", description="An acorn kissed by moonlight.", effect="Basic reagent for earth potions."),
    "goblin-bone": Item(id="goblin-bone", name="Goblin Bone", description="Rattling remains.", effect="Contains trace amounts of chaotic energy."),
    "vial": Item(id="vial", name="Empty Vial", description="Clear glass.", effect="Required to hold any liquid creation.")
})

fsm.set_context_data("prices", {"healing-potion": 30, "acorn": 5, "goblin-bone": 10, "vial": 2})
fsm.set_context_data("recipes", {"defense-potion": Recipe(result_id="defense-potion", ingredients={"acorn": 1, "goblin-bone": 1, "vial": 1})})

fsm.set_context_data("player", {"money": 100, "inventory": [ItemAmount(id="acorn", amount=3), ItemAmount(id="vial", amount=5)]})
fsm.set_context_data("witch", {"name": "Baba", "quest_given": False})

def resolve_item_id(input_id: str | None) -> str | None:
    if not input_id: return None
    items = fsm.get_context_data("items")
    clean = input_id.lower().strip().rstrip('s') # Gestione plurali
    for k, v in items.items():
        if clean == k or clean in v.name.lower(): return k
    return None

def print_inventory():
    p = fsm.get_context_data("player")
    it = fsm.get_context_data("items")
    res = f"\n{colorama.Fore.YELLOW}--- BAG (Gold: {p['money']}) ---{colorama.Fore.RESET}\n"
    for s in p["inventory"]:
        res += f" * {it[s.id].name}: {s.amount}\n"
    return res

@fsm.define_state(
    state_key="GREETING",
    prompt_template="You are Baba. Identify intent: BUY, BREW, IDENTIFY, END, or TALK (for quest/chat).",
    transitions={"BREW": "Brew", "BUY": "Buy", "IDENTIFY": "Iden", "TALK": "Talk", "END": "Exit"}
)
async def greeting_state(fsm, **kwargs):
    witch = fsm.get_context_data("witch")
    if not kwargs.get('will_transition', False):
        return "Baba: *Cackle*... I smell sulfur and potential. What brings you to my hut? I sell reagents, brew potions, and identify artifacts."

    ns = fsm.get_next_state()
    if not witch["quest_given"] and ns == "TALK": fsm.set_next_state("QUEST_OFFER")
    if ns == "IDENTIFY": return "Baba: 1 gold coin to reveal secrets. What shall I inspect?"
    if ns == "BREW": return "Baba: I can brew a Defense Potion if you have an acorn, a bone, and a vial. Shall we?"
    if ns == "BUY": return "Baba: My shelves are full. I have vials (2g), acorns (5g), bones (10g) and healing potions (30g)."
    return "Baba: Speak quickly, time is bubbling away."

@fsm.define_state(
    state_key="IDENTIFY",
    prompt_template="Identify item. Extract 'item_id'. If user cancels, return null.",
    transitions={"GREETING": "Back", "IDENTIFY": "Loop"},
    response_model=ActionResponseModel
)
async def identify_state(fsm, response, **kwargs):
    player = fsm.get_context_data("player")
    item_id = resolve_item_id(response.item_id)
    if not item_id:
        fsm.set_next_state("GREETING")
        return "Baba: Changed your mind? Hmph. Don't waste my sight."
    if player["money"] < 1:
        fsm.set_next_state("GREETING")
        return "Baba: No gold, no wisdom. Get out."

    player["money"] -= 1
    item = fsm.get_context_data("items")[item_id]
    fsm.set_next_state("GREETING")
    return f"Baba: *Peers into a crystal*... Ah, the {item.name}. {item.effect} Anything else?"

@fsm.define_state(
    state_key="BUY",
    prompt_template="User wants to buy. Extract 'item_id' and 'amount'.",
    transitions={"BUY_OK": "Success", "GREETING": "Back", "BUY": "Retry"},
    response_model=ActionResponseModel
)
async def buy_state(fsm, response, **kwargs):
    item_id = resolve_item_id(response.item_id)
    if not item_id: return "Baba: I don't sell that! Look at my shelves: vials, potions, bones, and acorns."

    player = fsm.get_context_data("player")
    price = fsm.get_context_data("prices").get(item_id, 99) * response.amount

    if player["money"] < price:
        fsm.set_next_state("GREETING")
        return f"Baba: {price} gold? You're a beggar! Come back when you're rich."

    player["money"] -= price
    # Update inventory
    found = False
    for s in player["inventory"]:
        if s.id == item_id:
            s.amount += response.amount
            found = True
            break
    if not found: player["inventory"].append(ItemAmount(id=item_id, amount=response.amount))

    fsm.set_next_state("BUY_OK")
    return f"Baba: Fine. Take your {item_id}. Keep the change, I don't want your filth."

@fsm.define_state(state_key="BUY_OK", prompt_template="Done", transitions={"GREETING": "Back"})
async def buy_ok(fsm, **kwargs): return "Baba: Do you need more, or can I go back to my brew?"

@fsm.define_state(
    state_key="BREW",
    prompt_template="User wants to brew 'defense-potion'. Extract 'item_id'.",
    transitions={"GREETING": "Cancel", "BREW_OK": "Success", "BREW": "Retry"},
    response_model=ActionResponseModel
)
async def brew_state(fsm, response, **kwargs):
    item_id = resolve_item_id(response.item_id)
    if item_id != "defense-potion": return "Baba: I only brew Defense Potions for travelers!"

    recipe = fsm.get_context_data("recipes")[item_id]
    player = fsm.get_context_data("player")

    # Check ingredients
    for ing_id, count in recipe.ingredients.items():
        if next((s.amount for s in player["inventory"] if s.id == ing_id), 0) < count:
            return f"Baba: You're missing the {ing_id}! Check your bag."

    # Consume ingredients
    fsm.set_next_state("BREW_OK")
    for ing_id, count in recipe.ingredients.items():
        for i, s in enumerate(player["inventory"]):
            if s.id == ing_id:
                s.amount -= count
                if s.amount <= 0: del player["inventory"][i]

    player["inventory"].append(ItemAmount(id=item_id, amount=1))
    return "Baba: *Stirs the pot vigorously*... There! A fresh Defense Potion. Careful, it's hot."

@fsm.define_state(state_key="BREW_OK", prompt_template="Done", transitions={"GREETING": "Back"})
async def brew_ok(fsm, **kwargs): return "Baba: My cauldron is still warm. Want another?"

@fsm.define_state(state_key="QUEST_OFFER", prompt_template="Offer quest for 'Magic Mushroom'.", transitions={"GREETING": "Done"})
async def quest_offer_state(fsm, response, **kwargs):
    fsm.get_context_data("witch")["quest_given"] = True
    return f"Baba: {response}\n{colorama.Fore.CYAN}[QUEST: THE MOONGLOW MUSHROOM]{colorama.Fore.RESET}"

@fsm.define_state(state_key="END", prompt_template="Bye", transitions={})
async def end_state(fsm, **kwargs): return "Baba: Begone! My hut needs to stretch its legs."

async def main():
    client = GeminiOpenAIWrapper(api_key=api_key)
    print(f"{colorama.Fore.MAGENTA}--- Baba's Hut ---{colorama.Fore.RESET}")
    print(f"{colorama.Fore.MAGENTA}Type !inventory to see your gold and items.{colorama.Fore.RESET}")

    # Avvio automatico
    run_state = await fsm.run_state_machine(client, user_input="START")
    print(f"{colorama.Fore.GREEN}{run_state.response}{colorama.Fore.RESET}")

    while not fsm.is_completed():
        user_input = input(f"{colorama.Fore.BLUE}Traveler{colorama.Fore.RESET}: ").strip()
        if not user_input: continue
        if user_input.startswith("!"):
            if "inventory" in user_input: print(print_inventory())
            continue

        run_state = await fsm.run_state_machine(client, user_input=user_input)
        if run_state:
            print(f"{colorama.Fore.GREEN}{run_state.response}{colorama.Fore.RESET}")

if __name__ == "__main__":
    asyncio.run(main())