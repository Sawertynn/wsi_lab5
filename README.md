# WSI LAB 5 - sztuczna sieć neuronowa

Zespół:
- Monika Jung
- Tomasz Kurzela

Prowadzący:
- Katarzyna Nałęcz-Charkiewicz

# Zadanie

Cel zadania polega na implementacji perceptronu dwuwarstwowego oraz nauczeniu go reprezentowania zadanej funkcji f(x), opisującej rozkład Laplace'a. Funkcja ta jest dana wzorem:

$$
f(x) = \frac{1}{2b}e^{-\frac{|x-\mu|}{b}}
$$

Zakres x: [-8, 8]
Wartości $\mu$ i b: $\mu$ = 0, b = 1

Zatem funkcja to: $f(x) = \frac{1}{2} e^{-|x|}$

Zadanie realizowane jest w zespołach dwuosobowych.


## Kroki do wykonania

1. Zaimplementuj perceptron dwuwarstwowy, który będzie reprezentował funkcję f(x) dla zadanego zakresu x oraz wartości  $\mu$ i b.
2. Zbadaj jakość aproksymacji, obliczając Mean Squared Error (MSE) oraz Mean Absolute Error (MAE) między wartościami rzeczywistymi funkcji a wartościami przewidywanymi przez sieć.
3. Przedstaw wykres funkcji rzeczywistej oraz funkcji przewidywanej przez sieć.
4. Zbadaj, jak liczba neuronów w warstwie ukrytej wpływa na jakość aproksymacji, zmieniając jej wartość i porównując wyniki.

## Wskazówki

- Użyj funkcji aktywacji sigmoidalnej w warstwie ukrytej i metody gradientowej do znajdowania wag sieci.
- Zadbaj o odpowiedni dobór współczynnika uczenia oraz liczby iteracji uczenia.


## Architektura sieci

Model został zaimplementowany w języku Python przy użyciu biblioteki `numpy`. Sieć składa się z:

1. **Warstwa wejściowa** – przyjmuje jedną wartość wejściową \( x \),
2. **Warstwa ukryta** – zawiera `hidden_size` neuronów, każdy z aktywacją sigmoidalną,
3. **Warstwa wyjściowa** – składa się z jednego neuronu (regresja), bez funkcji aktywacji.

Uczenie odbywa się za pomocą **propagacji wstecznej (backpropagation)** i **gradient descent**, z funkcją błędu **MSE (Mean Squared Error)**.


## Implementacja

- Wagi pierwszej warstwy inicjalizowane są losowo z przedziału [-1/sqrt(n), 1/sqrt(n)],
- Wagi drugiej warstwy początkowo ustawiono na zero,
- Dla stabilności uczenia zastosowano **normalizację danych wejściowych i wyjściowych** (średnia = 0, odchylenie standardowe = 1),
- Funkcja aktywacji `sigmoid` oraz jej pochodna zostały zaimplementowane ręcznie,
- Zaimplementowano metodę `train()`, która wykonuje propagację w przód i wstecz przez określoną liczbę epok.


## Testy i wyniki

Dla każdego wybranego rozmiaru perceptron był testowany 5 razy, aby zminimalizować udział losowości przy przy tworzeniu sieci na początku.

Wyniki błędów ze wszystkich prób:

MSE
| size | min | max |
| -- | ------- | -------- |
|  4 | 1.51e-05| 1.67e-03 |
|  7 | 2.14e-05| 2.27e-04 |
| 10 | 3.11e-04| 1.16e-02 |
| 13 | 7.68e-05| 1.68e-04 |
| 20 | 1.77e-04| 1.99e-04 |


MAE
| size | min | max |
| -- | ------- | -------- |
|  4 | 1.51e-05 1.67e-03 |
|  7 | 2.14e-05 2.27e-04 |
| 10 | 3.11e-04 1.16e-02 |
| 13 | 7.68e-05 1.68e-04 |
| 20 | 1.77e-04 1.99e-04 |


### Wykresy (dla najlepszego wyniku)

![](plots/normal_size=4.png)
![](plots/normal_size=7.png)
![](plots/normal_size=10.png)
![](plots/normal_size=13.png)
![](plots/normal_size=20.png)


### Porównanie z nieznormalizowanym

Przy okazji przetestowaliśmy sieć przy trenowaniu bez normalizacji.
Dla niewielu neuronów wyniki są dość dobre, ale im więcej tym większa szansa na problemy z wyliczaniem, i i model jest nieprzydatny.

MSE
| size | min | max |
| -- | ------- | -------- |
|  4 | 1.09e-04 | 2.01e-04 |
|  7 | 3.20e-04 | 6.27e-03 |
| 10 | nan | nan |
| 13 | nan | nan |
| 20 | nan | nan |

MAE
| size | min | max |
| -- | ------- | -------- |
|  4 | 1.09e-04 | 2.01e-04 |
|  7 | 3.20e-04 | 6.27e-03 |
| 10 | nan | nan |
| 13 | nan | nan |
| 20 | nan | nan |


# Podsumowanie

Normalizacja jest bardzo ważna, bez niej przy większych warstwach ukrytych warstwach jest ryzyko na problemy z liczeniem.

Więcej neuronów dawało lepszy wynik w najgorszym wypadku i niewiele gorszy w wypadku najlepszym.


