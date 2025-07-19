# LLM - uczenie - AI Engineering

## Fazy przetwarzania modelu

### Wstępny trening modelu LLM ( Pre-Training)

#### Stan początkowy - niewytrenowany model

Przed wytrenowaniem model LLM to pusta sieć neuronowa. Jej parametry ( wagi połączeń pomiędzy neuronami ) są
zainicjowane losowymi wartościami.

Nie posiada wiedzy o języku, strukturze, gramatyce, nie zna żadnych słów.
Zadanie jakiegokolwiek pytania do takiego modelu skutkowałoby wygenerowaniem wielu losowych znaków.

```
+--------------------------+
|  Wejście (Tokeny/Embedingi) |
+--------------------------+
             |             
             |  (Chaos Losowych Połączeń)
             V             
+--------------------------+
|       Warstwa 1          |
|    (Puste Wagi/Brak Wzorców) |
+--------------------------+
             |             
             |  (Szum, Brak Organizacji)
             V             
+--------------------------+
|       Warstwa 2          |
|    (Brak Semantycznego Zrozumienia) |
+--------------------------+
             |             
             |  (...)      
             V             
+--------------------------+
|       Warstwa N          |
|    (Brak Wyuczonych Relacji) |
+--------------------------+
             |             
             |             
             V             
+--------------------------+
| Wyjście (Losowe Przewidywania) |
+--------------------------+
```
#### Architektura LLM - Transformer

W kontekście LLM, Transformer jest architekturą, która pozwala modelowi przetwarzać i rozumieć sekwencje danych (takich jak tekst) w sposób wysoce równoległy i efektywny, uchwytując złożone zależności kontekstowe. Zasadniczo, jest to "silnik", który pobiera numeryczne reprezentacje słów i przekształca je w reprezentacje, z których można generować nowy tekst lub odpowiadać na pytania.

Został on opisany w dokumencie [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) , oraz [Czym jest i jak
działa Transformer (sieć neuronowa)](https://miroslawmamczur.pl/czym-jest-i-jak-dziala-transformer-siec-neuronowa/

Diagram architektury transformera wygląda następująco:

![transformer.jpg](img/transformer.jpg)

##### Słownik tokenów i macierz Input Embeddings

Pierwszą warstwą architektury Transformera jest macierz Input Embedding. Jej powstanie musi zostać zostać poprzedzone 
stworzeniem jeszcze bardziej pierwotnej struktury - słownika tokenów. 

**Słownik tokenów** - jest to struktura której zadaniem jest przetworzenie tekstu na format zrozumiały dla przetwarzania numerycznego.  

W procesie tokenizacji każde zdanie ze zbioru uczącego przetwarzane jest na jednostki:

Jako przykład niech posłuży zdanie: "Natura stwarza najpiękniejsze obrazy"

    * "Natura"
    * "stwarza"
    * "najpiękniejsze"
    * "obrazy"

  Następnie każdy token otrzymuje unikalny identyfikator liczbowy. Identyfikatory są stałe podczas całego procesu uczenia.

  * "Natura" (ID: 201)
  * "stwarza" (ID: 315)
  * "najpiękniejsze" (ID: 489)
  * "obrazy" (ID: 522)

W tym kroku model uczy się swojego słownika tokenów. Oznacza to, że zostają zidentyfikowane wszystkie unikalne słowa 
i pod-słowa, które będzie rozpoznawał.

Pierwszą warstwą architektury transformera jest **macierz Input Embedings** powstająca bezpośrednio po kroku tworzenia 
słownika tokenów. Jest to specjalna numeryczna reprezentacja słownika. Posiada tyle wierszy, 
   ile jest  
   elementów w słowniku. Do każdego tokena przypisany jest wektor liczb stałoprzecinkowych. Inicjalnie posiada on 
   losowe wartości, które będą aktywnie optymalizowane podczas całego procesu uczenia. Są one parametrami 
   modelu. Dzięki nim model uczy się znaczenia słów. Liczba kolumn wektora embeddingu jest hyperparametrem 
   przypisanym do modelu.  Często jest to 768, ale mogą to być inne wartości, np. 1024.

| Token            | ID  | Embedding (4 liczby)       |
|------------------|-----|----------------------------|
| "Natura"         | 201 | [0.12, -0.87, 0.45, 0.33]  |
| "stwarza"        | 315 | [-0.56, 0.91, -0.22, 0.77] |
| "najpiękniejsze" | 489 | [0.03, 0.65, -0.44, -0.19] |
| "obrazy"         | 522 | [0.88, -0.11, 0.29, -0.73] |


