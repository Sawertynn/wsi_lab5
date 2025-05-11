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
