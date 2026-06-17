# M2 vs M2b Examples

This file highlights representative tokens where M2b improves on M2,
where M2 is correct and M2b is wrong, and where both are wrong differently.

## M2b correct, M2 wrong

### Sentence 1 — Je sens qu' entre ça et les films de médecins et scientifiques fous que nous avons déjà vus , nous pourrions emprunter un autre chemin pour l' origine .

- Token: `qu'` (id=3)
- Gold: head=21, label=mark
- M2: head=5, label=case
- M2b: head=21, label=mark
- Automatic note: long-distance, function word, wrong head, wrong label

### Sentence 1 — Je sens qu' entre ça et les films de médecins et scientifiques fous que nous avons déjà vus , nous pourrions emprunter un autre chemin pour l' origine .

- Token: `ça` (id=5)
- Gold: head=21, label=obl:mod
- M2: head=2, label=obj
- M2b: head=21, label=obl:mod
- Automatic note: long-distance, function word, wrong head, wrong label

### Sentence 1 — Je sens qu' entre ça et les films de médecins et scientifiques fous que nous avons déjà vus , nous pourrions emprunter un autre chemin pour l' origine .

- Token: `films` (id=8)
- Gold: head=5, label=conj
- M2: head=2, label=obl:mod
- M2b: head=5, label=conj
- Automatic note: coordination, wrong head, wrong label

### Sentence 1 — Je sens qu' entre ça et les films de médecins et scientifiques fous que nous avons déjà vus , nous pourrions emprunter un autre chemin pour l' origine .

- Token: `scientifiques` (id=12)
- Gold: head=10, label=conj
- M2: head=10, label=amod
- M2b: head=10, label=conj
- Automatic note: coordination, wrong label

### Sentence 1 — Je sens qu' entre ça et les films de médecins et scientifiques fous que nous avons déjà vus , nous pourrions emprunter un autre chemin pour l' origine .

- Token: `fous` (id=13)
- Gold: head=12, label=amod
- M2: head=10, label=conj
- M2b: head=12, label=amod
- Automatic note: wrong head, wrong label

### Sentence 1 — Je sens qu' entre ça et les films de médecins et scientifiques fous que nous avons déjà vus , nous pourrions emprunter un autre chemin pour l' origine .

- Token: `que` (id=14)
- Gold: head=18, label=obj
- M2: head=16, label=obj
- M2b: head=18, label=obj
- Automatic note: long-distance, function word, wrong head

### Sentence 1 — Je sens qu' entre ça et les films de médecins et scientifiques fous que nous avons déjà vus , nous pourrions emprunter un autre chemin pour l' origine .

- Token: `avons` (id=16)
- Gold: head=18, label=aux:tense
- M2: head=8, label=acl:relcl
- M2b: head=18, label=aux:tense
- Automatic note: function word, wrong head, wrong label

## M2 correct, M2b wrong

### Sentence 1 — Je sens qu' entre ça et les films de médecins et scientifiques fous que nous avons déjà vus , nous pourrions emprunter un autre chemin pour l' origine .

- Token: `nous` (id=20)
- Gold: head=21, label=nsubj
- M2: head=21, label=nsubj
- M2b: head=21, label=iobj
- Automatic note: function word, wrong label

### Sentence 2 — On pourra toujours parler à propos d' Averroès de " décentrement de le Sujet " .

- Token: `"` (id=15)
- Gold: head=11, label=punct
- M2: head=11, label=punct
- M2b: head=2, label=punct
- Automatic note: long-distance, wrong head

### Sentence 3 — « Il a été largement démontré que la population civile de le territoire non autonome de le Sahara occidental est l' objet de diverses atteintes à les droits humains , comme la détention arbitraire , les coups et les tortures » , écrit l' ONG internationale , implantée dans 35 pays , citée par l' agence de presse sahraouie .

- Token: `autonome` (id=15)
- Gold: head=13, label=amod
- M2: head=13, label=amod
- M2b: head=9, label=amod
- Automatic note: wrong head

### Sentence 3 — « Il a été largement démontré que la population civile de le territoire non autonome de le Sahara occidental est l' objet de diverses atteintes à les droits humains , comme la détention arbitraire , les coups et les tortures » , écrit l' ONG internationale , implantée dans 35 pays , citée par l' agence de presse sahraouie .

- Token: `Sahara` (id=18)
- Gold: head=13, label=nmod
- M2: head=13, label=nmod
- M2b: head=9, label=nmod
- Automatic note: long-distance, wrong head

### Sentence 3 — « Il a été largement démontré que la population civile de le territoire non autonome de le Sahara occidental est l' objet de diverses atteintes à les droits humains , comme la détention arbitraire , les coups et les tortures » , écrit l' ONG internationale , implantée dans 35 pays , citée par l' agence de presse sahraouie .

- Token: `,` (id=52)
- Gold: head=53, label=punct
- M2: head=53, label=punct
- M2b: head=48, label=punct
- Automatic note: wrong head

### Sentence 3 — « Il a été largement démontré que la population civile de le territoire non autonome de le Sahara occidental est l' objet de diverses atteintes à les droits humains , comme la détention arbitraire , les coups et les tortures » , écrit l' ONG internationale , implantée dans 35 pays , citée par l' agence de presse sahraouie .

- Token: `citée` (id=53)
- Gold: head=48, label=conj
- M2: head=48, label=conj
- M2b: head=45, label=acl
- Automatic note: long-distance, coordination, wrong head, wrong label

### Sentence 4 — 1er : début de la présidence lituanienne de l' Organisation pour la Sécurité et la Coopération en Europe ( OSCE ) ( jusqu' à le 31 décembre ) , avant l' Irlande ( 2012 ) et l' Ukraine ( 2013 ) .

- Token: `Ukraine` (id=38)
- Gold: head=32, label=conj
- M2: head=32, label=conj
- M2b: head=3, label=conj
- Automatic note: long-distance, coordination, wrong head

## Both wrong differently

### Sentence 1 — Je sens qu' entre ça et les films de médecins et scientifiques fous que nous avons déjà vus , nous pourrions emprunter un autre chemin pour l' origine .

- Token: `nous` (id=15)
- Gold: head=18, label=nsubj
- M2: head=16, label=nsubj
- M2b: head=18, label=iobj
- Automatic note: function word, wrong head, wrong label

### Sentence 1 — Je sens qu' entre ça et les films de médecins et scientifiques fous que nous avons déjà vus , nous pourrions emprunter un autre chemin pour l' origine .

- Token: `vus` (id=18)
- Gold: head=8, label=acl:relcl
- M2: head=16, label=xcomp
- M2b: head=5, label=acl:relcl
- Automatic note: long-distance, clausal dependency, wrong head, wrong label

### Sentence 2 — On pourra toujours parler à propos d' Averroès de " décentrement de le Sujet " .

- Token: `décentrement` (id=11)
- Gold: head=4, label=obl:arg
- M2: head=6, label=nmod
- M2b: head=8, label=nmod
- Automatic note: long-distance, wrong head, wrong label

### Sentence 3 — « Il a été largement démontré que la population civile de le territoire non autonome de le Sahara occidental est l' objet de diverses atteintes à les droits humains , comme la détention arbitraire , les coups et les tortures » , écrit l' ONG internationale , implantée dans 35 pays , citée par l' agence de presse sahraouie .

- Token: `objet` (id=22)
- Gold: head=6, label=csubj:pass
- M2: head=9, label=nmod
- M2b: head=6, label=ccomp
- Automatic note: long-distance, wrong head, wrong label

### Sentence 3 — « Il a été largement démontré que la population civile de le territoire non autonome de le Sahara occidental est l' objet de diverses atteintes à les droits humains , comme la détention arbitraire , les coups et les tortures » , écrit l' ONG internationale , implantée dans 35 pays , citée par l' agence de presse sahraouie .

- Token: `détention` (id=33)
- Gold: head=25, label=nmod
- M2: head=9, label=conj
- M2b: head=25, label=conj
- Automatic note: long-distance, wrong head, wrong label

### Sentence 3 — « Il a été largement démontré que la population civile de le territoire non autonome de le Sahara occidental est l' objet de diverses atteintes à les droits humains , comme la détention arbitraire , les coups et les tortures » , écrit l' ONG internationale , implantée dans 35 pays , citée par l' agence de presse sahraouie .

- Token: `coups` (id=37)
- Gold: head=33, label=conj
- M2: head=9, label=nmod
- M2b: head=25, label=conj
- Automatic note: long-distance, coordination, wrong head, wrong label

