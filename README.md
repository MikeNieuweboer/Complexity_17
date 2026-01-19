# Complexity_17
Group assignment of group 17 for the 2026 Complexity course

## Idea's / Ideeën
* *spin die web maakt*: [paper](https://www.cell.com/current-biology/fulltext/S0960-9822(21)01270-7?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0960982221012707%3Fshowall%3Dtrue)  
* *scheur door matriaal*: [paper](https://sci-hub.box/10.1103/PhysRevE.74.016118)
* *self-regenerating Neural Cellular Automata*: [paper/demo](https://distill.pub/2020/growing-ca/?ref=https://githubhelp.com)
  - De auteur traint een neuraal netwerk doormiddel van backpropagation, en zo vanuit een enkele cell ontstaat een figuur (bv. van een salamander).
  - Wij zouden kunnen proberen om een versimpelde versie hiervan te doen:
    1. Ipv backpropagation kunnen we door middel van een simpel evolutionair algoritme een klein neuraal netwerkje trainen voor eenvoudige vormen, zoals cirkels of vierkanten (ipv salamanders)
    2. Als we eenmaal dit werkende hebben kunnen we kijken naar hoe bestand de pictogrammen zijn voor perturbaties. Dus we zetten bijvoorbeeld 0.x van de cellen in de grid willekeurig op leeg en kijken hoe het neurale netwerk probeert om het originele pictogram te regeneren.
    3. Waarschijnlijk is er een kritiek punt vanaf waar het genereren moeizamer gaat/faalt. Vervolgens kunnen we kijken wat voor gedrag er ontstaat. Toen ik even wat probeerde bij de live demo op de website, zag ik dat er soms kopieën ontstaan, of het pictogram kan bijvoorbeeld gewoonweg verdwijnen. Hopelijk vinden we dan iets van een power-law ergens.
* *Lenia: Biology of Artificial Life*: [paper](https://content.wolfram.com/sites/13/2019/10/28-3-1.pdf)
  - Eigenlijk wat er gebeurt als je een CA overal continuous maakt ipv discrete.
### *Application Domain*
*Korte Uitleg*
*Computational Model*: CA, etc.
*Complexity Phenomenon*: [Dit soort phenomenen](https://www.wikiwand.com/en/Complex_systems)

***
