# M2b vs M3 Examples

This file highlights representative tokens where M3 improves on M2b,
where M2b is correct and M3 is wrong, and where both are wrong differently.

## M3 correct, M2b wrong

### Sentence 1 — Je sens qu' entre ça et les films de médecins et scientifiques fous que nous avons déjà vus , nous pourrions emprunter un autre chemin pour l' origine .

- Token: `nous` (id=15)
- Gold: head=18, label=nsubj
- M2b: head=18, label=iobj
- M3: head=18, label=nsubj
- Automatic note: function word, wrong label

### Sentence 1 — Je sens qu' entre ça et les films de médecins et scientifiques fous que nous avons déjà vus , nous pourrions emprunter un autre chemin pour l' origine .

- Token: `nous` (id=20)
- Gold: head=21, label=nsubj
- M2b: head=21, label=iobj
- M3: head=21, label=nsubj
- Automatic note: function word, wrong label

### Sentence 2 — On pourra toujours parler à propos d' Averroès de " décentrement de le Sujet " .

- Token: `décentrement` (id=11)
- Gold: head=4, label=obl:arg
- M2b: head=8, label=nmod
- M3: head=4, label=obl:arg
- Automatic note: long-distance, wrong head, wrong label

### Sentence 2 — On pourra toujours parler à propos d' Averroès de " décentrement de le Sujet " .

- Token: `"` (id=15)
- Gold: head=11, label=punct
- M2b: head=2, label=punct
- M3: head=11, label=punct
- Automatic note: long-distance, wrong head

### Sentence 3 — « Il a été largement démontré que la population civile de le territoire non autonome de le Sahara occidental est l' objet de diverses atteintes à les droits humains , comme la détention arbitraire , les coups et les tortures » , écrit l' ONG internationale , implantée dans 35 pays , citée par l' agence de presse sahraouie .

- Token: `autonome` (id=15)
- Gold: head=13, label=amod
- M2b: head=9, label=amod
- M3: head=13, label=amod
- Automatic note: wrong head

### Sentence 3 — « Il a été largement démontré que la population civile de le territoire non autonome de le Sahara occidental est l' objet de diverses atteintes à les droits humains , comme la détention arbitraire , les coups et les tortures » , écrit l' ONG internationale , implantée dans 35 pays , citée par l' agence de presse sahraouie .

- Token: `Sahara` (id=18)
- Gold: head=13, label=nmod
- M2b: head=9, label=nmod
- M3: head=13, label=nmod
- Automatic note: long-distance, wrong head

### Sentence 3 — « Il a été largement démontré que la population civile de le territoire non autonome de le Sahara occidental est l' objet de diverses atteintes à les droits humains , comme la détention arbitraire , les coups et les tortures » , écrit l' ONG internationale , implantée dans 35 pays , citée par l' agence de presse sahraouie .

- Token: `détention` (id=33)
- Gold: head=25, label=nmod
- M2b: head=25, label=conj
- M3: head=25, label=nmod
- Automatic note: long-distance, wrong label

## M2b correct, M3 wrong

### Sentence 1 — Je sens qu' entre ça et les films de médecins et scientifiques fous que nous avons déjà vus , nous pourrions emprunter un autre chemin pour l' origine .

- Token: `,` (id=19)
- Gold: head=5, label=punct
- M2b: head=5, label=punct
- M3: head=8, label=punct
- Automatic note: long-distance, wrong head

### Sentence 2 — On pourra toujours parler à propos d' Averroès de " décentrement de le Sujet " .

- Token: `à` (id=5)
- Gold: head=4, label=advmod
- M2b: head=4, label=advmod
- M3: head=6, label=case
- Automatic note: function word, wrong head, wrong label

### Sentence 2 — On pourra toujours parler à propos d' Averroès de " décentrement de le Sujet " .

- Token: `propos` (id=6)
- Gold: head=5, label=fixed
- M2b: head=5, label=fixed
- M3: head=4, label=obl:arg
- Automatic note: wrong head, wrong label

### Sentence 2 — On pourra toujours parler à propos d' Averroès de " décentrement de le Sujet " .

- Token: `Averroès` (id=8)
- Gold: head=5, label=obl:arg
- M2b: head=5, label=obl:arg
- M3: head=6, label=nmod
- Automatic note: wrong head, wrong label

### Sentence 3 — « Il a été largement démontré que la population civile de le territoire non autonome de le Sahara occidental est l' objet de diverses atteintes à les droits humains , comme la détention arbitraire , les coups et les tortures » , écrit l' ONG internationale , implantée dans 35 pays , citée par l' agence de presse sahraouie .

- Token: `pays` (id=51)
- Gold: head=48, label=obl:arg
- M2b: head=48, label=obl:arg
- M3: head=48, label=obl:mod
- Automatic note: wrong label

### Sentence 7 — Ils ne citent pas son nom , parce que depuis les institutions on n' attaque pas un membre de la famille royale , mais c' est à lui qu' ils s' en prennent .

- Token: `attaque` (id=15)
- Gold: head=3, label=advcl
- M2b: head=3, label=advcl
- M3: head=3, label=conj
- Automatic note: long-distance, clausal dependency, wrong label

### Sentence 7 — Ils ne citent pas son nom , parce que depuis les institutions on n' attaque pas un membre de la famille royale , mais c' est à lui qu' ils s' en prennent .

- Token: `lui` (id=28)
- Gold: head=3, label=conj
- M2b: head=3, label=conj
- M3: head=15, label=conj
- Automatic note: long-distance, coordination, function word, wrong head

## Both wrong differently

### Sentence 1 — Je sens qu' entre ça et les films de médecins et scientifiques fous que nous avons déjà vus , nous pourrions emprunter un autre chemin pour l' origine .

- Token: `vus` (id=18)
- Gold: head=8, label=acl:relcl
- M2b: head=5, label=acl:relcl
- M3: head=12, label=acl:relcl
- Automatic note: long-distance, clausal dependency, wrong head

### Sentence 4 — 1er : début de la présidence lituanienne de l' Organisation pour la Sécurité et la Coopération en Europe ( OSCE ) ( jusqu' à le 31 décembre ) , avant l' Irlande ( 2012 ) et l' Ukraine ( 2013 ) .

- Token: `début` (id=3)
- Gold: head=0, label=root
- M2b: head=1, label=appos
- M3: head=1, label=nmod
- Automatic note: wrong ROOT, wrong head, wrong label

### Sentence 4 — 1er : début de la présidence lituanienne de l' Organisation pour la Sécurité et la Coopération en Europe ( OSCE ) ( jusqu' à le 31 décembre ) , avant l' Irlande ( 2012 ) et l' Ukraine ( 2013 ) .

- Token: `Irlande` (id=32)
- Gold: head=3, label=nmod
- M2b: head=3, label=conj
- M3: head=10, label=nmod
- Automatic note: long-distance, wrong head, wrong label

### Sentence 8 — Il y en a qui croient toujours qu' ils peuvent jouir de la vie en toute tranquillité de conscience et commettre telle ou telle indélicatesse , telle ou telle grossièreté , en toute sécurité , parce que leur coeur serait comme les montres : 100 % Waterproof .

- Token: `croient` (id=6)
- Gold: head=3, label=acl:relcl
- M2b: head=4, label=ccomp
- M3: head=4, label=advcl:cleft
- Automatic note: clausal dependency, wrong head, wrong label

### Sentence 8 — Il y en a qui croient toujours qu' ils peuvent jouir de la vie en toute tranquillité de conscience et commettre telle ou telle indélicatesse , telle ou telle grossièreté , en toute sécurité , parce que leur coeur serait comme les montres : 100 % Waterproof .

- Token: `montres` (id=43)
- Gold: head=47, label=obl:mod
- M2b: head=21, label=obj
- M3: head=10, label=obl:mod
- Automatic note: long-distance, wrong head, wrong label

### Sentence 10 — Pour lui , je crois vraiment que l' essentiel est ailleurs , dans cette contribution à la déstabilisation de l' autorité spirituelle de le pape que vous évoquez .

- Token: `contribution` (id=15)
- Gold: head=11, label=conj
- M2b: head=10, label=obl:arg
- M3: head=5, label=obl:mod
- Automatic note: long-distance, coordination, wrong head, wrong label

