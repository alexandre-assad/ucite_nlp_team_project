# M0p vs M1 Examples

This file highlights representative tokens where M1 improves on M0p,
where M0p is correct and M1 is wrong, and where both are wrong differently.

## M1 correct, M0p wrong

### Sentence 1 — Je sens qu' entre ça et les films de médecins et scientifiques fous que nous avons déjà vus , nous pourrions emprunter un autre chemin pour l' origine .

- Token: `qu'` (id=3)
- Gold: head=21, label=mark
- M0p: head=18, label=mark
- M1: head=21, label=mark
- Automatic note: long-distance, function word, wrong head

### Sentence 1 — Je sens qu' entre ça et les films de médecins et scientifiques fous que nous avons déjà vus , nous pourrions emprunter un autre chemin pour l' origine .

- Token: `films` (id=8)
- Gold: head=5, label=conj
- M0p: head=18, label=nsubj
- M1: head=5, label=conj
- Automatic note: coordination, wrong head, wrong label

### Sentence 1 — Je sens qu' entre ça et les films de médecins et scientifiques fous que nous avons déjà vus , nous pourrions emprunter un autre chemin pour l' origine .

- Token: `scientifiques` (id=12)
- Gold: head=10, label=conj
- M0p: head=8, label=conj
- M1: head=10, label=conj
- Automatic note: coordination, wrong head

### Sentence 1 — Je sens qu' entre ça et les films de médecins et scientifiques fous que nous avons déjà vus , nous pourrions emprunter un autre chemin pour l' origine .

- Token: `vus` (id=18)
- Gold: head=8, label=acl:relcl
- M0p: head=2, label=ccomp
- M1: head=8, label=acl:relcl
- Automatic note: long-distance, clausal dependency, wrong head, wrong label

### Sentence 1 — Je sens qu' entre ça et les films de médecins et scientifiques fous que nous avons déjà vus , nous pourrions emprunter un autre chemin pour l' origine .

- Token: `,` (id=19)
- Gold: head=5, label=punct
- M0p: head=21, label=punct
- M1: head=5, label=punct
- Automatic note: long-distance, wrong head

### Sentence 1 — Je sens qu' entre ça et les films de médecins et scientifiques fous que nous avons déjà vus , nous pourrions emprunter un autre chemin pour l' origine .

- Token: `pourrions` (id=21)
- Gold: head=2, label=ccomp
- M0p: head=18, label=conj
- M1: head=2, label=ccomp
- Automatic note: long-distance, clausal dependency, wrong head, wrong label

### Sentence 3 — « Il a été largement démontré que la population civile de le territoire non autonome de le Sahara occidental est l' objet de diverses atteintes à les droits humains , comme la détention arbitraire , les coups et les tortures » , écrit l' ONG internationale , implantée dans 35 pays , citée par l' agence de presse sahraouie .

- Token: `population` (id=9)
- Gold: head=22, label=nsubj
- M0p: head=13, label=nsubj
- M1: head=22, label=nsubj
- Automatic note: long-distance, wrong head

## M0p correct, M1 wrong

### Sentence 3 — « Il a été largement démontré que la population civile de le territoire non autonome de le Sahara occidental est l' objet de diverses atteintes à les droits humains , comme la détention arbitraire , les coups et les tortures » , écrit l' ONG internationale , implantée dans 35 pays , citée par l' agence de presse sahraouie .

- Token: `,` (id=35)
- Gold: head=37, label=punct
- M0p: head=37, label=punct
- M1: head=25, label=punct
- Automatic note: wrong head

### Sentence 3 — « Il a été largement démontré que la population civile de le territoire non autonome de le Sahara occidental est l' objet de diverses atteintes à les droits humains , comme la détention arbitraire , les coups et les tortures » , écrit l' ONG internationale , implantée dans 35 pays , citée par l' agence de presse sahraouie .

- Token: `pays` (id=51)
- Gold: head=48, label=obl:arg
- M0p: head=48, label=obl:arg
- M1: head=48, label=obl:mod
- Automatic note: wrong label

### Sentence 3 — « Il a été largement démontré que la population civile de le territoire non autonome de le Sahara occidental est l' objet de diverses atteintes à les droits humains , comme la détention arbitraire , les coups et les tortures » , écrit l' ONG internationale , implantée dans 35 pays , citée par l' agence de presse sahraouie .

- Token: `sahraouie` (id=59)
- Gold: head=56, label=amod
- M0p: head=56, label=amod
- M1: head=58, label=amod
- Automatic note: wrong head

### Sentence 4 — 1er : début de la présidence lituanienne de l' Organisation pour la Sécurité et la Coopération en Europe ( OSCE ) ( jusqu' à le 31 décembre ) , avant l' Irlande ( 2012 ) et l' Ukraine ( 2013 ) .

- Token: `1er` (id=1)
- Gold: head=3, label=nummod
- M0p: head=3, label=nummod
- M1: head=0, label=root
- Automatic note: wrong ROOT, wrong head, wrong label

### Sentence 5 — Et pourtant , lors de sa première visite en Afrique subsaharienne , l' été 2007 , il montre les dents dans un discours humiliant pour les Africains , prononcé à l' Université Cheikh Anta Diop de Dakar .

- Token: `prononcé` (id=29)
- Gold: head=23, label=acl
- M0p: head=23, label=acl
- M1: head=18, label=advcl
- Automatic note: long-distance, clausal dependency, wrong head, wrong label

### Sentence 6 — Les spéculations autour de le match sont à leur paroxysme .

- Token: `autour` (id=3)
- Gold: head=2, label=advmod
- M0p: head=2, label=advmod
- M1: head=10, label=advmod
- Automatic note: wrong head

### Sentence 8 — Il y en a qui croient toujours qu' ils peuvent jouir de la vie en toute tranquillité de conscience et commettre telle ou telle indélicatesse , telle ou telle grossièreté , en toute sécurité , parce que leur coeur serait comme les montres : 100 % Waterproof .

- Token: `telle` (id=22)
- Gold: head=25, label=det
- M0p: head=25, label=det
- M1: head=21, label=obj
- Automatic note: function word, wrong head, wrong label

## Both wrong differently

### Sentence 1 — Je sens qu' entre ça et les films de médecins et scientifiques fous que nous avons déjà vus , nous pourrions emprunter un autre chemin pour l' origine .

- Token: `ça` (id=5)
- Gold: head=21, label=obl:mod
- M0p: head=8, label=obl:mod
- M1: head=21, label=nsubj
- Automatic note: long-distance, function word, wrong head, wrong label

### Sentence 2 — On pourra toujours parler à propos d' Averroès de " décentrement de le Sujet " .

- Token: `propos` (id=6)
- Gold: head=5, label=fixed
- M0p: head=4, label=obl:arg
- M1: head=4, label=obl:mod
- Automatic note: wrong head, wrong label

### Sentence 2 — On pourra toujours parler à propos d' Averroès de " décentrement de le Sujet " .

- Token: `"` (id=15)
- Gold: head=11, label=punct
- M0p: head=2, label=punct
- M1: head=6, label=punct
- Automatic note: long-distance, wrong head

### Sentence 3 — « Il a été largement démontré que la population civile de le territoire non autonome de le Sahara occidental est l' objet de diverses atteintes à les droits humains , comme la détention arbitraire , les coups et les tortures » , écrit l' ONG internationale , implantée dans 35 pays , citée par l' agence de presse sahraouie .

- Token: `objet` (id=22)
- Gold: head=6, label=csubj:pass
- M0p: head=6, label=obl:mod
- M1: head=6, label=ccomp
- Automatic note: long-distance, wrong label

### Sentence 3 — « Il a été largement démontré que la population civile de le territoire non autonome de le Sahara occidental est l' objet de diverses atteintes à les droits humains , comme la détention arbitraire , les coups et les tortures » , écrit l' ONG internationale , implantée dans 35 pays , citée par l' agence de presse sahraouie .

- Token: `coups` (id=37)
- Gold: head=33, label=conj
- M0p: head=25, label=conj
- M1: head=43, label=nsubj
- Automatic note: long-distance, coordination, wrong head, wrong label

### Sentence 3 — « Il a été largement démontré que la population civile de le territoire non autonome de le Sahara occidental est l' objet de diverses atteintes à les droits humains , comme la détention arbitraire , les coups et les tortures » , écrit l' ONG internationale , implantée dans 35 pays , citée par l' agence de presse sahraouie .

- Token: `tortures` (id=40)
- Gold: head=33, label=conj
- M0p: head=37, label=nmod
- M1: head=37, label=conj
- Automatic note: long-distance, coordination, wrong head, wrong label

