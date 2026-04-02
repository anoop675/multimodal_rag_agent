"""
Utility helpers.
"""


def pretty_print_recipe(recipe: dict):
    print("\n" + "═" * 60)
    print(f"  {recipe.get('title', 'Recipe').upper()}")
    print("═" * 60)
    print(f"  {recipe.get('description', '')}")
    # print(f"\n  Serves {recipe.get('servings','?')}  |  "
    #       f"Prep {recipe.get('prep_time_minutes','?')} min  |  "
    #       f"Cook {recipe.get('cook_time_minutes','?')} min")
    print("\n  INGREDIENTS\n  " + "─" * 40)
    for ing in recipe.get("ingredients", []):
        print(f" * {ing.get('quantity', '')}  {ing.get('item', '')}")
    print("\n  METHOD\n  " + "─" * 40)
    for step in recipe.get("steps", []):
        print(f"  {step.get('step', '')}. {step.get('instruction', '')}")
    tips = recipe.get("tips", [])
    if tips:
        print("\n  TIPS")
        for tip in tips:
            print(f" * {tip}")
    print("═" * 60 + "\n")
