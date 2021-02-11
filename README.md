# Traffic Sign Detection

**Student:** Nikola Maksimovic SW15/2016

**Asistent:** Dragan Vidakovic

**Definicija problema:** Potrebno je izvrsiti klasifikaciju saobracajnih znakova koji budu u vidokrugu kamere. Ulaz u sistem je frejm kamere, na osnovu kog se generise izlaz koji je informacija o klasi (tipu saobracajnog znaka) kao i tacnost kojom se predict-uje ta klasa na osnovu sadrzaja frejma.

**Motivacija:** Detekcija saobracajnih znakova se koristi prilikom kontrole i upravljanja polu-autonomnih vozila, a jednog dana ce se koristiti, sigurno i u potpuno autonomnim vozilima.

**Skup podataka:** Koristice se postojeci skup podataka [(link](https://drive.google.com/file/d/1AZeKw90Cb6GgamTBO3mvDdz6PjBwqCCt/view)) koji sadrzi 30000+ slika od 40+ klasa. Podaci ce biti izdeljeni na 3 grupe: train, test i validation u odnosu: 64:20:16.

**Metodologija:** Koristice se biblioteke _opencv_, _keras_ i _tensorflow_. Svaka slika iz skupa podataka ce biti predprocesirana (konvertovanje u _grayscale_, standardizacija osvetljenja i normalizacija). Takodje, vrsice se augmentacija, kako bi se izbegao _"overfitting"_ i kako bi se povecala generalizacija modela. _CNN_ model bi se sastojao od nekoliko konvolutivnih slojeva, _max-pooling_ i _Dense_ slojeva, kao i po jedan _Flatten_ i _Dropout_ sloj. 

**Evaluacija:** Metrike koje bi se koristile za evaluaciju _CNN_ bi bile accuracy i loss.
