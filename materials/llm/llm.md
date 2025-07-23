# LLM - uczenie - AI Engineering

### Architektura LLM - Transformer

W kontekście LLM, Transformer jest architekturą, która pozwala modelowi przetwarzać i rozumieć sekwencje danych (takich
jak tekst) w sposób wysoce równoległy i efektywny, uchwytując złożone zależności kontekstowe. Zasadniczo, jest to "
silnik", który pobiera numeryczne reprezentacje słów i przekształca je w reprezentacje, z których można generować nowy
tekst lub odpowiadać na pytania.

Został on opisany w dokumencie [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) , oraz [Czym jest i jak
działa Transformer (sieć neuronowa)](https://miroslawmamczur.pl/czym-jest-i-jak-dziala-transformer-siec-neuronowa/

Wyróżnić można trz główne topologie architektury Transformera. Wybór właściwej topologii zależy od rodzaju zadań,
które ma wykonywać model. Różnią się one tym które bloki **(Encoder/Decoder)** są w nich używane.

1. **Encoder Only**

![encoder.jpg](img/encoder.jpg)

Ten typ architektury składa się wyłącznie ze stosu komponentów **Encoder** . Występuje w nim wiele warstw Encoderów
nabudowanych jedna na drugiej.
Przy uczeniu każdy token może widzieć i czerpać kontekst ze wszystkich innych tokenów w sekwencji.

Modele tego rodzaju osiągają najlepsze wyniki w realizacji zadania rozumienia języka naturalnego **(NLU- Natural
Language Understanding)**

- Klasyfikacja tekstu (np. analiza sentymentu, wykrywanie spamu)
- Odpowiadanie na pytania do tekstu (najlepiej z wyborem)
- Rozpoznawanie nazwanych encji (NER) czyli np. wyszukiwanie nazw własnych, kolorów itp.
- Rozumienie relacji logicznych w tekście
- Ekstrakcja informacji
- Wyszukiwanie informacji

Modele tego typu nie są najlepsze w generowaniu dłuższych tekstów.

Najbardziej znany model z tego typu architekturą to **BERT (BIdeirectional Encoder Representations from Transformers)**

2. **Decoder Only**

![decoder.jpg](img/decoder.jpg)

W tym typie architektury występuje wyłącznie stos komponentów **Decoder** , które ułożone są w nabudowane na siebie
warstwy.

Model tego typu może czerpać kontekst tyko z tokenów które go poprzedzają w sekwencji. Nie może odnosić się do tokenów,
które występują po nim.

Uczy się przewidywać następny token na podstawie sekwencji poprzednich.

Modele tego typu są wykorzystywane głównie w zadaniach generowania języka naturalnego **(NLG - Natural Language
Generation)**

- Generowanie tekstu (eseje, artykuły, wiersze, kod)
- Tworzenie chatbotów
- Streszczenie

Ten rodzaj architektury wykorzystywany jest w modelach:

- GPT (GPT-1, GPT-2, GPT-3, GPT-4)
- LLaMA
- Mistral
- Gemma

3. **Encoder-Decoder**

![transformer.jpg](img/transformer.jpg)

Jest to pełna forma Transformera opisana w przywoływanym przeze mnie artykule [Attention Is All You Need]
(https://arxiv.org/pdf/1706.03762) . Łączy ona w sobie cechy obydwu opisanych wcześniej topologii. Wykorzystuje
zarówno warstwy komponentów **Encoder** jak i **Decoder**.

Architektura tego typu najlepiej nadaje się do zadań przekształcania sekwencji:

- Tłumaczenia tekstu (z jednego języka na inny)
- Streszczenia
- Parafrazy
- Konwersja danych do innego formatu

### Fazy przetwarzania modelu

#### Wstępny trening modelu LLM (Pre-Training) w architekturze Encoder-Only

#### Przebieg procesu trenowania modelu LLM

Przed wytrenowaniem model LLM to pusta sieć neuronowa. Jej parametry (wagi połączeń pomiędzy neuronami) są
zainicjowane losowymi wartościami.

Nie posiada wiedzy o języku, strukturze, gramatyce, nie zna żadnych słów.
Zadanie jakiegokolwiek pytania do takiego modelu skutkowałoby wygenerowaniem wielu losowych znaków.

Proces wstępnego uczenia modelu LLM (pre-training) jest złożony, spróbujmy jednak spojrzeć na niego z pewnego
oddalenia. Można wyróżnić następujące jego etapy:

1. Przygotowanie danych

- Zbieranie i czyszczenie - Gromadzi się biliony tokenów surowego tekstu z Internetu (książki, artykuły, Wikipedia,
  itp..)
- Tokenizacja i słownik - Na podstawie całego korpusu budowany jest słownik tokenów. Każdy token dostaje swoje ID.
  Tekst jest przekształcany w sekwencje ID tokenów.
- Formatowanie do Treningu - Dane są dzielone na mniejsze części/sekwencje o ustalonej maksymalnej długości (np. 512,
  768, 1024, itp..). Jeśli zdania są dłuższe podlegają podziałowi, jeśli krótsze - dopełnieniu.

2. Określenie hiperparametrów modelu

- Określenie hiperparametrów definiujących architekturę np: liczba warstw Encodera, liczba głowic uwagi, wymiar
  modelu (dmodel)
- Hiperparametry definiujące trening np: liczba sekwencji przetwarzanych równolegle, liczba kroków optymalizacji, itp...

3. Faza pretreningowa

- Losowa inicjalizacja wag (Encoder, głowice MLM, tablice embeddings)

- Główna pętla treningowa:

```
  * Pobieranie minipartii - Z ogromnego datasetu pobierana jest mini-partia sekwencji tekstowych.
  
  * Dynamiczne maskowanie (MLM) - W mini-partii, w locie niektóre tokeny są maskowane zgodnie ze strategią MLM.
  
  * Przepływ do Enkodera - Sekwencje przetwarzane są do **Final Input Embeddings** i podawane do stosu **Enkoderów**.
  
  * Przetwarzanie w Enkoderze - Enkoder przetwarza sekwencje, tworzy bogate, kontekstualizowane reprezentacje dla
    danego tokena.
  
  * Przekazanie do Głowic Predykcyjnych - Jest to etap przyuczania modelu do wykonywania działań, np: MLM - czyli
    odgadywanie zamaskowanych tokenów. W tym kroku wytworzone przez Enkodery reprezentacje przekazywane są do
    odpowiednich głowić (tu w zależności od zadań do których przeznaczony jest model, można wykorzystać różne głowice,
    np: MLM Head - dla zamaskowanych tokenów).
  
  * Generowanie wektorów logit i obliczanie Strat - Głowicie predykcyjne generują wektory **logit** a po zastosowaniu
    na nich odpowiednich funkcji - gotowych predykcji. Predykcje te porównywane są z odpowiedziami "prawdziwymi" (a
    więc niezamaskowanymi, wzorcowymi). Na podstawie tych porównań obliczana jest **łączna funkcja straty** .
  
  * Wsteczna propagacja - Informacja o stracie (gradienty) jest propagowana wstecz przez cały model, od głowic
    predykcyjnych, przez wszystkie warstwy Enkodera, aż do tablicy Embeddings. Te nowe, zaktualizowane wagi będą 
    brane pod uwagę przy przetwarzaniu kolejnych sekwencji i minipartii modelu.
  
  * Aktualizacja wag - Algorytm optymalizacji używa gradientów do dostosowywania wag w modelu.
```

* Monitorowanie postępów - Trening jest ciągle monitorowany, obserwuje się wartość funkcji straty (wraz z postępami
  powinna ona spadać), mierzy się wydajność zbiorów walidacyjnych.

4. Zakończenie pretrenowania

Po tym etapie otrzymujemy **Model Bazowy** – potężny, ogólny model językowy, który rozumie gramatykę, semantykę i kontekst,
ale nie jest jeszcze przystosowany do konkretnych zadań użytkownika.

W procesie uczenia modelu bazowego występuje pojęcie dynamicznego maskowania, czy nie jest to jedno z zadań 
użytkownika ? Okazuje się, że nie. Zadania 
pretreningowe, takie jak np. MLM (maskowanie), czy NSP (dogenerowywanie) nie są celem samym w sobie. To tylko środek 
do realizacji zadań użytkownika (a umiejętności ich realizacji model bazowy jeszcze nie posiada, ma pozyskać je 
dopiero w fazie uczenia potreningowego). 

Są to np: 
 
- Klasyfikacja sentymentu
- Odpowiedzi na pytania
- Streszczenia artykułów
- itp..

Należy zwrócić uwagę że w procesie uczenia, zbiór uczący podlega maskowaniu. W praktyce oznacza to, że w trakcie 
treningu losowo wybierany jest pewien procent tokenów (np. 15% ), a następnie te wybrane tokeny jest zamieniana na 
specjalny token _[MASK]_ lub zamieniana na losowy token ze słownika (ale o tym za chwilę). Te zmienione tokeny wędrują 
przez całą siatkę Enkoderów, więc również dla nich budowana jest ich reprezentacja kontekstowa. Fakt ten jest 
źródłem pewnej przypadłości (szczególnie wczesnych modeli) nazywanej _mask token toxicity_. Polega ona 
na tym, że model, który w danych uczących stosunkowo często napotyka token _[MASK]_ może uznać go jako "szczególnie 
ważny" i zacząć go nadmiernie używać w swoich odpowiedziach, szczególnie gdyby nie był ich zbyt pewny.
Aby zminimalizować to ryzyko stosuje się zaawansowane techniki maskowania np:

Po wylosowaniu 15% tokenów dzieli się je na trzy podgrupy:

1. 80% z nich jest faktycznie zastępowana tokenem _[MASK]_. To jest główne zsadanie, które zmusza model do uczenia 
   się kontekstu. Model uczy się że jest to token specjalny (jego embedding nie reprezentuje żadnego znaczenia w 
   języku) , którego wartość należy wywnioskować na podstawie 
   analizy jego bezpośredniego otoczenia.
2. 10% tokenów z tej grupy jest zastępowana losowym tokenem ze słownika. To uczy model, że nawet jeśli napotka 
   zwyczajnie wyglądające słowo nadal może być zmuszony do jego przewidywania na podstawie kontekstu.
3. 10% tokenów pozostaje niezmienionych. To uczy model, że czasami "prawdziwa odpowiedź" to słowo, które już widzi.

Ta strategia wprowadza element szumu i niepewności, ucząc że token _[MASK]_ nie jest jedynym "sygnałem braku", a 
także że musi polegać na kontekście, nawet jeśli napotka prawdziwe słowo.

Żeby lepiej wyobrazić sobie konsekwencje tej strategii rozpatrzmy dwa zdania:

``` Natura opona najpiękniejsze obrazy.```
``` Opona jest ważną częścią koła samochodowego. ```

W pierwszym przypadku słowo _opona_ jest przykładem tokenu maskującego, który został losowo wybrany ze słownika. 
Model będzie próbował budować reprezentację kontekstową wszystkich tokenów, uwzględniając słowo _opona_ jako 
zwyczajny element otoczenia.  Model zakoduje informację o tym że słowo _opona_ znajduje się w otoczeniu _Natura_, 
_najpiękniejsze_ i _obrazy_ . Wewnętrznie model będzie czuł, że słowo to wprowadza niespójność i że jest ono mało 
prawdopodobne w porównaniu do innych sekwencji, z którymi się spotkał np: _Opona jest ważną częścią koła samochodowego._
Wartości liczbowe określające związek tego słowa z bezpośrednim otoczeniem będą słabsze, przez co model uczy się, że 
taka kobinacja liczb jest nietypowa.

#### Słownik tokenów i macierz Input Embeddings

Zanim rozpocznie się faktyczny proces uczenia, konieczne jest przygotowanie dwóch powiązanych ze sobą struktur
pomocniczych - **Słownika tokenów** oraz wektora **Input
Embedding**. Jej powstanie musi zostać poprzedzone
stworzeniem jeszcze bardziej pierwotnej struktury - słownika tokenów.

**Słownik tokenów** - jest to struktura której zadaniem jest przetworzenie tekstu na format zrozumiały dla przetwarzania
numerycznego.

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

Pierwszą warstwą architektury transformera jest **macierz Input Embeddings** powstająca bezpośrednio po kroku tworzenia
słownika tokenów. Jest to specjalna numeryczna reprezentacja słownika. Posiada tyle wierszy,
ile jest  
elementów w słowniku. Do każdego tokena przypisany jest wektor liczb stałoprzecinkowych. Inicjalnie posiada on
losowe wartości, które będą aktywnie optymalizowane podczas całego procesu uczenia. Są one parametrami
modelu. Dzięki nim model uczy się znaczenia słów. Liczba kolumn wektora embeddingu jest hyperparametrem
przypisanym do modelu. Często jest to 768, ale mogą to być inne wartości, np. 1024.

| Token            | ID  | Embedding (4 liczby)       |
|------------------|-----|----------------------------|
| "Natura"         | 201 | [0.12, -0.87, 0.45, 0.33]  |
| "stwarza"        | 315 | [-0.56, 0.91, -0.22, 0.77] |
| "najpiękniejsze" | 489 | [0.03, 0.65, -0.44, -0.19] |
| "obrazy"         | 522 | [0.88, -0.11, 0.29, -0.73] |

#### Tablica Positional Embeddings

Przed rozpoczęciem treningu konieczne jest stworzenie globalnej ** tablicy positional embeddings ** . Jest globalnym
zbiorem wektorów dla wszystkich możliwych pozycji jakie model może napotkać. Tablica ta ma tyle wierszy ile tokenów
może mieć największa sekwencja przetwarzana przez ten model (a więc np. 512, 1024, 2048, ... , itd...). Dla każdej
pozycji przechowywany jest wektor embedding , o długości zgodnej z długością embeddingów określonych w parametrach
modelu(np. 768, 1024). Wektory input i positional mają te same wymiary,co pozawla na ich wzajemne sumowanie.

Przy założeniu że nasz model może przetwarzać sekwencje o maksymalnej długości 8 tokenów i określono dla
niego _wymiar_embeddingu = 4_ , to inicjalna tablica embeddingów pozycyjnych mogłaby wyglądać następująco:

| Pozycja | Wektor Embeddingu Pozycyjnego (PE) - Wymiar 4 |
|---------|-----------------------------------------------|
| 0       | [0.01, 0.02, 0.03, 0.04]                      |
| 1       | [0.05, 0.06, 0.07, 0.08]                      |
| 2       | [0.09, 0.10, 0.11, 0.12]                      |
| 3       | [0.13, 0.14, 0.15, 0.16]                      |
| 4       | [0.17, 0.18, 0.19, 0.20]                      |
| 5       | [0.21, 0.22, 0.23, 0.24]                      |
| 6       | [0.25, 0.26, 0.27, 0.28]                      |
| 7       | [0.29, 0.30, 0.31, 0.32]                      |
| ...     | ... (aż do max\_seq\_length - 1)              |

Zatem posługując się identyfikatorem pozycji tokenu jak łącznikiem możemy skojarzyć ze sobą element z tablicy
**Input Embeddings** z elementem z tablicy **Positional Embeddings** .

| Pozycja | Token            | Wektor Kodowania Pozycyjnego (PE) - Wymiar 4 |
|---------|------------------|----------------------------------------------|
| 0       | "Natura"         | [0.01, 0.02, 0.03, 0.04]                     |
| 1       | "stwarza"        | [0.05, 0.06, 0.07, 0.08]                     |
| 2       | "najpiękniejsze" | [0.09, 0.10, 0.11, 0.12]                     |
| 3       | "obrazy"         | [0.13, 0.14, 0.15, 0.16]                     |

Dla każdego elementu mamy do dyspozycji dwa wektory embeddings, po jednym z każdej tabeli.

Teraz dla każdego tokenu dokonujemy sumowania embeddingów i w ten sposób powstanie **Final Input Embedding**, który ma
zakodowaną informację zarówno o znaczeniu semantycznym tokenu, jak i jego pozycji w sekwencji. **Final Input
Embedding** jest to struktura, która będzie wykorzystywana w dalszym procesie uczenia.

| Pozycja | Token            | ID  | Input Embedding (4 liczby) | Wektor Kodowania Pozycyjnego (PE) - Wymiar 4 | Finalny Input Embedding (Sumowany) |
|---------|------------------|-----|----------------------------|----------------------------------------------|------------------------------------|
| 0       | "Natura"         | 201 | [0.12, -0.87, 0.45, 0.33]  | [0.01, 0.02, 0.03, 0.04]                     | [0.13, -0.85, 0.48, 0.37]          |
| 1       | "stwarza"        | 315 | [-0.56, 0.91, -0.22, 0.77] | [0.05, 0.06, 0.07, 0.08]                     | [-0.51, 0.97, -0.15, 0.85]         |
| 2       | "najpiękniejsze" | 489 | [0.03, 0.65, -0.44, -0.19] | [0.09, 0.10, 0.11, 0.12]                     | [0.12, 0.75, -0.33, -0.07]         |
| 3       | "obrazy"         | 522 | [0.88, -0.11, 0.29, -0.73] | [0.13, 0.14, 0.15, 0.16]                     | [1.01, 0.03, 0.44, -0.57]          |

Wektor wynikowy jest unikalny dla konkretnego słowa na konkretnej pozycji. Dalsze warstwy transformera, a zwłaszcza
mechanizm uwagi są sieciami neuronowymi, które podczas treningu uczą się jak interpretować te wzbogacone wektory.

Finalny wektor **Input Embedding** jest skonstruowany w taki sposób, aby słowa o podobnym znaczeniu miały podobne
wektory (czyli znalazły się blisko siebie w przestrzeni wielowymiarowej). Wymiar tej przestrzeni jest ograniczony
wymiarem wektora **Input Embedding**. Jeśli ma on długość 768 to znaczy że każde słowo, lub token reprezentowane
jest jako punkt w 768 wymiarowej przestrzeni wektorowej.

Dla lepszego zrozumienia można posłużyć się analogią do smaku potrawy:

Składa się z wielu składników, które są ze sobą wymieszane. Nie "rozdzielasz" ich w ustach, ale twój mózg uczy się
rozpoznawać poszczególne smaki i ich kombinacje. Podobnie Transformer uczy się, że pewne kombinacje wartości w
wektorze (które wynikają z sumy V_słowo i PE) sygnalizują, że "to jest rzeczownik na początku zdania", a inne
kombinacje, że "to jest czasownik w środku zdania".

#### Sposób działania Encodera ( w modelu Encoder-Only)

Jest to ta część architektury Transformera która odpowiada za uczenie modelu. Składa się on z kilku
dwuelementowych bloków (warstw). Zwykle jest ich kilka - od 6 do 12.
Każda taka warstwa przetwarza dane i przekazuje je do następnej, stopniowo budując coraz bardziej złożone i abstrakcyjne
rozumienie wejściowej sekwencji.

![encoder.jpg](img/encoder.jpg)

##### Jak działa pojedyncza Warstwa Enkodera?

Pojedyncza warstwa encodera składa się z dwóch wyspecjalizowanych podbloków

* Multi-Head Self-Attention (Wielogłowicowa Self-Attention)
* Feed-Forward Network (Sieć Przewijająca do Przodu)

**Multi-Head Self-Attention**

Głowica uwagi ( **Attention Head** ) to kluczowy komponent architektury **Transformer**. Jest to filtr, lub
perspektywa, która jest wykorzystywana do analizowania relacji pomiędzy tokenami w sekwencji. Wiele ( **Attention
Head** ) składa się na mechanizm (**Multi Head**). Dzięki temu, że każda z głowic , w procesie uczenia  
specjalizuje się w jakimś specjalnym aspekcie powiązań pomiędzy tokenami.

Z każdą głowicą związane są trzy unikalne dla niej macierze Wag, które podlegają procesowi uczenia. Są to macierze
**(Wq, Wk, Wv)**. Na początku uczenia są one inicjalizowane losowymi wartościami, które będą podlegać modyfikacjom (na
tym polega ta specjalizacja). Macierze wag są unikalne dla każdej głowicy, ale są one takie same dla wszystkich
tokenów przetwarzanych przez tę głowicę.

Proces przetwarzania dla każdej głowicy odbywa się według stałej sekwencji:

1. Każda głowica otrzymuje wektory **Final Input Embedding** dla wszystkich tokenów składających się na
   przetwarzaną sekwencję.
2. Głowica, dla każdego tokena w sekwencji wykonuje mnożenie własnych macierzy wag **(Wq,Wk,Wv)** z każdym z wektorem  
   **Final Input Embedding** dla każdego z tokenów, w ten sposób powstają trzy wektory pomocnicze Q(Query), K(Key),
   Value(V) dla każdego tokena w sekwencji. W standardowej konfiguracji długość wektorów Q,K,V oblicza się dzieląc
   długość wektora **Final Input Embedding**  przez liczbę głowic. dk=dv=dmodel/h (liczba głowic),

Przykład:

    dk=dv=dmodel 768(długość wektora FIE)/8(liczba głowic) = 96.   

4. **Mechanizm uwagi**
    * Macierz **Q** aktualnie przetwarzanego tokenu jest porównywana z wektorem **K** każdego
      tokenu w sekwencji. Realizowane jest to przez wykonanie iloczynu skalarnego pomiędzy tymi wektorami. Wynik tej
      operacji prowadzi do wyznaczenia współczynników uwagi i określa stopień dopasowania pomiędzy tokenami. Daje to
      odpowiedź na pytanie jak istotny jest każdy inny token w sekwencji dla zrozumienia
      sensu aktualnie przetwarzanego tokena. Jeśli **K** innego tokena pasuje do **Q** bieżącego tokenta to znaczy
      że on jest "ważny".
    * Na podstawie wyznaczonych współczynników uwagi, oraz wektora wartości **V** generowana jest ważona suma **V**
      wszystkich tokenów wchodzących w skład sekwencji. Powstają więc wektory **V1,V2,V3,...,Vn**. Jest to wynik pracy
      głowicy. Jest to nowa, bardziej
      kontekstualizowana reprezentacja dla przetwarzanego tokena.

Przykład dla tokena "Natura":

        Wyjście z Głowicy 1 dla "Natura": [v_1_1, v_1_2, ..., v_1_96] (wymiar 96)
   
        Wyjście z Głowicy 2 dla "Natura": [v_2_1, v_2_2, ..., v_2_96] (wymiar 96)
        ...
        
        Wyjście z Głowicy h dla "Natura": [v_h_1, v_h_2, ..., v_h_96] (wymiar 96)

4. Wyniki ze wszystkich głowic dla danego tokena są konkatenowane, tworzony jest jeden bardzo długi wektor, który
   zawiera wartości liczbowe zawierające syntezę wszysktich różnych perspektyw z poszczególnych głowic.

Kontynuacja przykładu dla tokena "Natura":

        Skonkatenowany wektor dla "Natura" = [v_1_1, ..., v_1_96, v_2_1, ..., v_2_96, ..., v_h_1, ..., v_h_96]
        
        Łączny wymiar po konkatenacji wynosi h * dv = dmodel (768)

5. Następnie ten długi wektor przepuszczany jest przez finalną, nauczalną macierz wag **W0 (Output Projection)**.
   Macierz **W0 (Output Projection)** jest trenowana aby jak najlepiej łączyć projekcje. Macierz a jest parametrem
   modelu używanym w komponencie **Multi-Head Attention** do finalnego ukształtowania jego wyjścia (nie jest
   natomiast samym wyjściem)
6. Obliczane jest wyjście z komponentu **Multi-Head Attention**. Jest to wynik mnożenia macierzy **V** przez macierz
   **W0**. Wynikiem mnożenia jest finalny wektor wyjściowy z Multi-Head Attention dla danego tokena ma długość
   dmodel, czyli zgodną z długością wejściowego wektora **Final Input Embedding*.

Macierz wyjściowa z komponentu **Multi-Head Attention** poddawana jest dalszemu przetwarzaniu przez komponent ("Add
& Norm") na które składają się operacje:

1. Połączenia Resztkowego (Add): Sumuje wejście i wyjście komponentu **Multi-Head Attention**
2. Normalizację Warstwową (Norm): Wykonuje normalizację wyniku.

Trafia do kolejnego komponentu w obrębie Encodera - **Feed-Forward-Network**.

**Feed-Forward-Network**
Jest to komponent uzupełniający w stosunku do **Multi-Head Self-Attention**. Podczas gdy mechanizm uwagi skupia się na
relacjach pomiędzy tokenami, **FFN** skupia się na reprezentacji pojedynczego tokena. Architektonicznie jest to
dwuwarstwowa sieć neuronowa z zastosowaną nieliniową funkcją aktywacji pomiędzy warstwami. Celem przetwarzania
przez ten komponent jest odkrycie subtelnych i złożonych cech z reprezentacji pochodzących z mechanizmu uwagi.

Wynik działania **FFN** poddawany jest operacjom **Add & Norm** i kierowany do kolejnej z warstw **Encodera**.

Przetwarzanie w kolejnych warstwach odbywa się w sposób analogiczny do opisanego wyżej. Należy jednak zwrócić uwagę na
jeden bardzo istotny fakt. Wejściem dla pierwszej warstwy jest wektor  **Input Embedding** . To on jest podstawą
wykonywanych w tej warstwie obliczeń. W przypadku kolejnych encoderów wejściem do warstwy jest wektor wyjściowy z
warstwy poprzedniej. Dzięki temu następuje coraz większe doprecyzowanie kontekstu.

Po przetworzeniu przez wszystkie warstwy, dla każdego tokena w oryginalnej sekwencji otrzymujemy finalną
reprezentację w postaci wektora. Jest to więc macierz o wymiarach (_długość_sekwencjixdmodel_), kazdy wiersz to wektor
o długości _dmodel(np. 768)_ odpowiadający jednemu tokenowi.

Ta finalna reprezentacja będzie podstawą do kolejnego etapu treningu - wykonywania zadań pretreningowych.

#### Sposób działania Decodera (w modelu Decoder-Only)

![decoder.jpg](img/decoder.jpg)