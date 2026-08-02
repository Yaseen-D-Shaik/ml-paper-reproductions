PEOPLE = [
    "Christopher", "Roberto", "Penelope", "Maria", "Andrew", "Pierro",
    "Christine", "Francesca", "Arthur", "Emilio", "Margaret", "Gina",
    "Victoria", "Lucia", "James", "Marco", "Jennifer", "Angela",
    "Charles", "Tomaso", "Colin", "Alfonse", "Charlotte", "Sophia"
]

RELATIONSHIPS = [
    "father", "mother", "husband", "wife", "son", "daughter", 
    "uncle", "aunt", "brother", "sister", "nephew", "niece"
]

# Raw relational dataset mapping (Person1, Relation) -> [Person2 options]
FAMILY_DATA = {
    # --- English Family Branch ---
    ("Christopher", "husband"): ["Penelope"],
    ("Penelope", "wife"): ["Christopher"],
    ("Andrew", "husband"): ["Christine"],
    ("Christine", "wife"): ["Andrew"],
    ("Arthur", "husband"): ["Margaret"],
    ("Margaret", "wife"): ["Arthur"],
    ("James", "husband"): ["Victoria"],
    ("Victoria", "wife"): ["James"],
    ("Charles", "husband"): ["Jennifer"],
    ("Jennifer", "wife"): ["Charles"],
    
    ("Christopher", "father"): ["Victoria", "Arthur"],
    ("Penelope", "mother"): ["Victoria", "Arthur"],
    ("Andrew", "father"): ["James", "Charles"],
    ("Christine", "mother"): ["James", "Charles"],
    ("Victoria", "mother"): ["Colin", "Charlotte"],
    ("James", "father"): ["Colin", "Charlotte"],

    ("Victoria", "daughter"): ["Christopher", "Penelope"],
    ("Arthur", "son"): ["Christopher", "Penelope"],
    ("James", "son"): ["Andrew", "Christine"],
    ("Charles", "son"): ["Andrew", "Christine"],
    ("Colin", "son"): ["Victoria", "James"],
    ("Charlotte", "daughter"): ["Victoria", "James"],

    ("Victoria", "sister"): ["Arthur"],
    ("Arthur", "brother"): ["Victoria"],
    ("James", "brother"): ["Charles"],
    ("Charles", "brother"): ["James"],
    ("Colin", "brother"): ["Charlotte"],
    ("Charlotte", "sister"): ["Colin"],

    ("Arthur", "uncle"): ["Colin", "Charlotte"],
    ("Margaret", "aunt"): ["Colin", "Charlotte"],
    ("Andrew", "uncle"): ["Colin", "Charlotte"],
    ("Christine", "aunt"): ["Colin", "Charlotte"],

    ("Colin", "uncle"): ["Arthur", "Charles"],
    ("Colin", "aunt"): ["Margaret", "Jennifer"],
    ("Charlotte", "uncle"): ["Arthur", "Charles"],
    ("Charlotte", "aunt"): ["Margaret", "Jennifer"],

    ("Colin", "nephew"): ["Arthur", "Margaret", "Charles", "Jennifer"],
    ("Charlotte", "niece"): ["Arthur", "Margaret", "Charles", "Jennifer"],
    ("James", "nephew"): ["Arthur", "Margaret"],
    ("Charles", "nephew"): ["Victoria", "James"],
    ("Victoria", "niece"): ["Charles", "Jennifer"],

    # --- Italian Family Branch ---
    ("Roberto", "husband"): ["Maria"],
    ("Maria", "wife"): ["Roberto"],
    ("Pierro", "husband"): ["Francesca"],
    ("Francesca", "wife"): ["Pierro"],
    ("Emilio", "husband"): ["Gina"],
    ("Gina", "wife"): ["Emilio"],
    ("Marco", "husband"): ["Lucia"],
    ("Lucia", "wife"): ["Marco"],
    ("Tomaso", "husband"): ["Angela"],
    ("Angela", "wife"): ["Tomaso"],

    ("Roberto", "father"): ["Lucia", "Emilio"],
    ("Maria", "mother"): ["Lucia", "Emilio"],
    ("Pierro", "father"): ["Marco", "Angela"],
    ("Francesca", "mother"): ["Marco", "Angela"],
    ("Lucia", "mother"): ["Alfonse", "Sophia"],
    ("Marco", "father"): ["Alfonse", "Sophia"],

    ("Lucia", "daughter"): ["Roberto", "Maria"],
    ("Emilio", "son"): ["Roberto", "Maria"],
    ("Marco", "son"): ["Pierro", "Francesca"],
    ("Angela", "daughter"): ["Pierro", "Francesca"],
    ("Alfonse", "son"): ["Lucia", "Marco"],
    ("Sophia", "daughter"): ["Lucia", "Marco"],

    ("Lucia", "sister"): ["Emilio"],
    ("Emilio", "brother"): ["Lucia"],
    ("Marco", "brother"): ["Angela"],
    ("Angela", "sister"): ["Marco"],
    ("Alfonse", "brother"): ["Sophia"],
    ("Sophia", "sister"): ["Alfonse"],

    ("Emilio", "uncle"): ["Alfonse", "Sophia"],
    ("Gina", "aunt"): ["Alfonse", "Sophia"],

    ("Alfonse", "uncle"): ["Emilio", "Angela"],
    ("Alfonse", "aunt"): ["Gina", "Tomaso"],
    ("Sophia", "uncle"): ["Emilio", "Angela"],
    ("Sophia", "aunt"): ["Gina", "Tomaso"],

    ("Alfonse", "nephew"): ["Emilio", "Gina"],
    ("Sophia", "niece"): ["Emilio", "Gina"],
}