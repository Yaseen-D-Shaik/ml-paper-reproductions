"""Family tree triples for the Rumelhart 1986 reproduction."""

# Add family tree data here.
PEOPLE= ["Christopher","Roberto","Penelope", "Maria", "Andrew", "pierro",
         "christine","francesca","arthur","emilio","margaret","gina","james",
         "Marco","Victoria","lucia","charies","tomaso","jennifer",
         "Angela","colin","Alfonso","charlotte","sophia"]


RELATIONSHIPS= ["father", "mother","husband", "wife", "son", "daughter", 
                "uncle", "aunt", "brother", "sister", "nephew","niece"]


TRIPLES= {
    # -------------------------
    # ENGLISH FAMILY
    # -------------------------

    # Christopher (gen 1, branch 1)
    ("Christopher", "wife"):        ["Penelope"],
    ("Christopher", "son"):         ["Arthur"],
    ("Christopher", "daughter"):    ["Victoria"],

    # Penelope (gen 1, branch 1)
    ("Penelope", "husband"):        ["Christopher"],
    ("Penelope", "son"):            ["Arthur"],
    ("Penelope", "daughter"):       ["Victoria"],

    # Andrew (gen 1, branch 2)
    ("Andrew", "wife"):             ["Christine"],
    ("Andrew", "son"):              ["James"],
    ("Andrew", "daughter"):         ["Jennifer"],

    # Christine (gen 1, branch 2)
    ("Christine", "husband"):       ["Andrew"],
    ("Christine", "son"):           ["James"],
    ("Christine", "daughter"):      ["Jennifer"],

    # Arthur (gen 2, branch 1) — no children, married Margaret
    ("Arthur", "father"):           ["Christopher"],
    ("Arthur", "mother"):           ["Penelope"],
    ("Arthur", "wife"):             ["Margaret"],
    ("Arthur", "sister"):           ["Victoria"],
    ("Arthur", "nephew"):           ["Colin"],
    ("Arthur", "niece"):            ["Charlotte"],

    # Margaret (gen 2, married into branch 1)
    ("Margaret", "husband"):        ["Arthur"],
    ("Margaret", "nephew"):         ["Colin"],
    ("Margaret", "niece"):          ["Charlotte"],

    # Victoria (gen 2, links both branches)
    ("Victoria", "father"):         ["Christopher"],
    ("Victoria", "mother"):         ["Penelope"],
    ("Victoria", "husband"):        ["James"],
    ("Victoria", "brother"):        ["Arthur"],
    ("Victoria", "son"):            ["Colin"],
    ("Victoria", "daughter"):       ["Charlotte"],

    # James (gen 2, branch 2)
    ("James", "father"):            ["Andrew"],
    ("James", "mother"):            ["Christine"],
    ("James", "wife"):              ["Victoria"],
    ("James", "sister"):            ["Jennifer"],
    ("James", "son"):               ["Colin"],
    ("James", "daughter"):          ["Charlotte"],

    # Jennifer (gen 2, branch 2) — no children
    ("Jennifer", "father"):         ["Andrew"],
    ("Jennifer", "mother"):         ["Christine"],
    ("Jennifer", "husband"):        ["Charles"],
    ("Jennifer", "brother"):        ["James"],

    # Charles (gen 2, married into branch 2)
    ("Charles", "wife"):            ["Jennifer"],
    ("Charles", "nephew"):          ["Colin"],
    ("Charles", "niece"):           ["Charlotte"],

    # Colin (gen 3)
    ("Colin", "father"):            ["James"],
    ("Colin", "mother"):            ["Victoria"],
    ("Colin", "uncle"):             ["Arthur", "Charles"],
    ("Colin", "aunt"):              ["Jennifer", "Margaret"],
    ("Colin", "sister"):            ["Charlotte"],

    # Charlotte (gen 3)
    ("Charlotte", "father"):        ["James"],
    ("Charlotte", "mother"):        ["Victoria"],
    ("Charlotte", "uncle"):         ["Arthur", "Charles"],
    ("Charlotte", "aunt"):          ["Jennifer", "Margaret"],
    ("Charlotte", "brother"):       ["Colin"],

    # -------------------------
    # ITALIAN FAMILY (isomorphic)
    # Roberto=Christopher, Maria=Penelope
    # Pierro=Andrew, Francesca=Christine
    # Emilio=Arthur, Gina=Margaret
    # Lucia=Victoria, Marco=James
    # Angela=Jennifer, Tomaso=Charles
    # Alfonse=Colin, Sophia=Charlotte
    # -------------------------

    # Roberto (gen 1, branch 1)
    ("Roberto", "wife"):            ["Maria"],
    ("Roberto", "son"):             ["Emilio"],
    ("Roberto", "daughter"):        ["Lucia"],

    # Maria (gen 1, branch 1)
    ("Maria", "husband"):           ["Roberto"],
    ("Maria", "son"):               ["Emilio"],
    ("Maria", "daughter"):          ["Lucia"],

    # Pierro (gen 1, branch 2)
    ("Pierro", "wife"):             ["Francesca"],
    ("Pierro", "son"):              ["Marco"],
    ("Pierro", "daughter"):         ["Angela"],

    # Francesca (gen 1, branch 2)
    ("Francesca", "husband"):       ["Pierro"],
    ("Francesca", "son"):           ["Marco"],
    ("Francesca", "daughter"):      ["Angela"],

    # Emilio (gen 2, branch 1) — no children, married Gina
    ("Emilio", "father"):           ["Roberto"],
    ("Emilio", "mother"):           ["Maria"],
    ("Emilio", "wife"):             ["Gina"],
    ("Emilio", "sister"):           ["Lucia"],
    ("Emilio", "nephew"):           ["Alfonse"],
    ("Emilio", "niece"):            ["Sophia"],

    # Gina (gen 2, married into branch 1)
    ("Gina", "husband"):            ["Emilio"],
    ("Gina", "nephew"):             ["Alfonse"],
    ("Gina", "niece"):              ["Sophia"],

    # Lucia (gen 2, links both branches)
    ("Lucia", "father"):            ["Roberto"],
    ("Lucia", "mother"):            ["Maria"],
    ("Lucia", "husband"):           ["Marco"],
    ("Lucia", "brother"):           ["Emilio"],
    ("Lucia", "son"):               ["Alfonse"],
    ("Lucia", "daughter"):          ["Sophia"],

    # Marco (gen 2, branch 2)
    ("Marco", "father"):            ["Pierro"],
    ("Marco", "mother"):            ["Francesca"],
    ("Marco", "wife"):              ["Lucia"],
    ("Marco", "sister"):            ["Angela"],
    ("Marco", "son"):               ["Alfonse"],
    ("Marco", "daughter"):          ["Sophia"],

    # Angela (gen 2, branch 2) — no children
    ("Angela", "father"):           ["Pierro"],
    ("Angela", "mother"):           ["Francesca"],
    ("Angela", "husband"):          ["Tomaso"],
    ("Angela", "brother"):          ["Marco"],

    # Tomaso (gen 2, married into branch 2)
    ("Tomaso", "wife"):             ["Angela"],
    ("Tomaso", "nephew"):           ["Alfonse"],
    ("Tomaso", "niece"):            ["Sophia"],

    # Alfonse (gen 3)
    ("Alfonse", "father"):          ["Marco"],
    ("Alfonse", "mother"):          ["Lucia"],
    ("Alfonse", "uncle"):           ["Emilio", "Tomaso"],
    ("Alfonse", "aunt"):            ["Angela", "Gina"],
    ("Alfonse", "sister"):          ["Sophia"],

    # Sophia (gen 3)
    ("Sophia", "father"):           ["Marco"],
    ("Sophia", "mother"):           ["Lucia"],
    ("Sophia", "uncle"):            ["Emilio", "Tomaso"],
    ("Sophia", "aunt"):             ["Angela", "Gina"],
    ("Sophia", "brother"):          ["Alfonse"],
}