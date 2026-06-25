# KL Annealing — resumo

Versão curta e prática. Para a derivação teórica completa, ver `kl_annealing.md`.

## O que é

A loss da VAE soma dois termos que puxam para lados opostos:

$$\text{loss} = R + \beta\,L$$

- **$R$ (reconstrução):** quer reconstruir bem a imagem.
- **$L$ (KL):** quer deixar o espaço latente parecido com $\mathcal{N}(0, I)$ — ou seja, organizado, contínuo e bem comportado.

Normalmente $\beta = 1$ (os dois termos pesam igual). **KL annealing** é só isto: colocar um peso $\beta$ no termo KL e fazê-lo **crescer aos poucos** ao longo do treino, em vez de já começar no máximo.

## Por que isso existe (a parte prática)

O problema que o annealing resolve é concreto: **sem ele, a VAE costuma "desligar" o espaço latente nas primeiras épocas e nunca mais recuperar.** Isso se chama *posterior collapse*. As razões, na prática:

**1. O atalho preguiçoso do otimizador.**
No início, o decoder ainda é ruim — não sabe transformar um código latente em imagem. Então, para baixar a loss, o caminho mais fácil **não** é aprender a reconstruir; é zerar o termo KL, fazendo o encoder devolver sempre o mesmo valor genérico ($\mu = 0$, $\sigma = 1$) e ignorar a entrada. O latente vira ruído inútil, e como não sobra gradiente para reativar aquelas dimensões, elas ficam "mortas" para sempre. Começar com $\beta \approx 0$ tira essa tentação da mesa enquanto o decoder ainda não está pronto.

**2. Aprender na ordem certa.**
É mais fácil **primeiro aprender a representar** (como um autoencoder comum) e **depois organizar** o espaço. Com $\beta = 0$ a rede povoa o latente com informação útil; quando $\beta$ sobe, ela só precisa *arrumar* um espaço que já faz sentido — em vez de tentar organizar o vazio.

**3. Estabilidade no começo.**
Com o encoder ainda aleatório, o termo KL pode ser grande e instável logo de cara (ainda mais com learning rate alto, que é justamente quando o collapse é mais provável). Entrar com peso pequeno e ir subindo deixa as primeiras épocas mais suaves.

**4. Custo quase zero.**
Não muda a arquitetura nem o objetivo final: em $\beta = 1$ você tem a VAE normal de volta. É só um "aquecimento" do regularizador — quando ajuda, ajuda muito; quando não precisa, não atrapalha.

Em uma frase: **a loss da VAE tem um mínimo trivial e ruim (latente colapsado) que é fácil demais de atingir cedo; o annealing adia a pressão do KL até a rede ter algo que valha a pena regularizar.**

## Como está no projeto

Schedule em degraus: $\beta$ começa em 0 e sobe **+0.2 a cada 5 épocas**, até chegar em 1.0.

| época | 1–4 | 5–9 | 10–14 | 15–19 | 20–24 | ≥ 25 |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: |
| β | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |

```python
KL_BETA_STEP = 0.2
def kl_beta(epoch, anneal_epochs, step=KL_BETA_STEP):
    if anneal_epochs <= 0:
        return 1.0                       # constante: sem annealing
    return min(1.0, step * (epoch // anneal_epochs))
```

Dois pontos práticos:

- A **avaliação usa sempre $\beta = 1$**, então a métrica de comparação é a loss "real". Assim um treino com annealing e um sem competem de forma justa.
- Nem todo treino sofre collapse — depende do dataset, da capacidade do decoder e do learning rate. Por isso o annealing entra como um **teste A/B**: `constant` ($\beta = 1$) **vs** `step0.2_per5ep`. Vence o de menor loss de teste, e esse segue para os próximos estágios do tuning. Se o seu caso não precisar, o `constant` ganha e tudo bem.

## Por que importa aqui

Esta VAE é o encoder/decoder da **Latent Diffusion**, que espera um latente parecido com $\mathcal{N}(0, I)$, denso e contínuo. Queremos a organização que o KL traz — mas **sem** o colapso que ele pode causar se vier com força cedo demais. O annealing entrega os dois: primeiro a rede aprende a representar, depois o latente é moldado para o formato que a difusão precisa.
