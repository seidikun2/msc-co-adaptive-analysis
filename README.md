# Análise de execução motora, imagética e pseudo-coadaptação

Projeto local em Python, com EDA por sujeito e avaliação dos classificadores. Os
CSV originais não são alterados. As configurações ficam em **`config.yaml`**.
Não é necessário GPU, banco de dados ou servidor web.

**Estado desta entrega:** fluxo validado com sinais sintéticos, inclusive geração
dos relatórios. Os quatro anexos reais excederam o limite de transferência de
32 MiB por arquivo da ferramenta; seu conteúdo integral ainda não foi processado
aqui. O cabeçalho fornecido informa 512 Hz, 16 canais EEG, dois auxiliares e
marcações OpenViBE. As posições da touca e a unidade física continuam pendentes.
Os exemplos gerados por `demo` são explicitamente sintéticos.

## 1. Instalar e inspecionar

Use Python **3.11 ou 3.12**, preferencialmente 3.12, em um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_analysis.py inspect --data /caminho/para/dados
```

No PowerShell, substitua a ativação por `.venv\Scripts\Activate.ps1`.

O comando `inspect` não exige conhecer unidade ou atenção. Ele lê cada arquivo,
confere timestamps, taxa e canais, e produz:

- `results/inventory.csv`: nomes, sujeito, sessão, bloco, duração, contagem dos
  códigos e erros de leitura. Pode servir como manifest editável para grupos.
- `results/markers.csv`: **todas** as marcações, com instante original, duração,
  código e nome padrão quando conhecido. Ajuda a conferir a sequência real.

`Epoch` no CSV OpenViBE é índice do buffer de aquisição, **não** a identidade da
tentativa. As tentativas são reconstruídas pelos eventos de atenção e cue.
`Event Date` tem prioridade sobre o timestamp da linha: por exemplo, um evento
em `0.0234241611` s não deve ser deslocado para o instante da amostra que o contém.
Eventos múltiplos separados por `:` e linhas contendo apenas eventos são aceitos.
Descontinuidades e amostras parcialmente faltantes geram erro; não são interpoladas.

## 2. Organização dos arquivos e grupos

A descoberta é recursiva. A convenção confirmada é:

```text
dados/
  grupo_opcional/
    EG102/
      motor_me_EG102_1.csv
      motor_mi_EG102_1.csv
      motor_mi_EG102_online_1a.csv
      motor_mi_EG102_online_1b.csv
      motor_mi_EG102_online_1c.csv
      motor_mi_EG102_online_1d.csv
      ... mesmas seis categorias nas sessões 2 a 6 ...
```

`me` significa execução; `mi` no nome é convertido para `im` nas tabelas.
`EG102_2` significa **sessão 2**, não o segundo bloco. Sujeito pode mudar.
São esperados 36 arquivos por sujeito; faltas aparecem no inventário do relatório.
O código não confunde `online_1a` com sessão/bloco offline.

**Não é necessário manifest nem inventory para executar.** Use diretamente `--data`;
a busca entra nas subpastas dos sujeitos, inclusive dentro de outras pastas.
`Organizacao_experimentos.csv` e `motor_16ch.csv` são sempre ignorados por
`inspect` e `run` (sem diferenciar maiúsculas/minúsculas). Para excluir outros
arquivos/pastas, configure no bloco `input` de `config.yaml`:

```yaml
exclude_globs: [backup, 'rascunhos/*', '*_config.csv']
```

Também é possível acrescentar exclusões sem editar YAML, repetindo `--exclude`:
`python run_analysis.py run --data /caminho/para/dados --exclude backup --exclude '*_config.csv'`.
O padrão não apaga arquivos; as exclusões usadas ficam em `config-used.yaml`.

Cada padrão corresponde ao nome ou ao caminho relativo à raiz `--data`, com `/`
como separador. Uma pasta correspondente exclui toda a sua árvore. CSVs restantes
fora da convenção continuam sendo sinalizados; não são descartados silenciosamente.
Os padrões personalizados se aplicam à busca automática. Os dois auxiliares
conhecidos também são ignorados se aparecerem em um manifest antigo. O terminal
mostra `Auxiliar ignorado (não é EEG)` para deixar essa decisão visível.

`Organizacao_experimentos.csv` descreve a organização experimental.
`motor_16ch.csv` descreve canais/eletrodos, não amostras: o formato fornecido é
`channel;channel_label;x;y;z`, com `Channel 1` a `Channel 16` e `Channel Ref`.
A linha de referência não é um 17º canal EEG. Esse arquivo pode fundamentar a
montagem após verificar unidade e sistema de coordenadas; não é carregado
automaticamente como EEG nem usado para confirmar a montagem provisória.

Se ainda aparecer a mensagem antiga `Use inspect e um manifest` citando esses
auxiliares, atualize seu checkout antes de executar (preserve alterações locais;
se houver conflito, não use reset):

```bash
git switch main
git pull --ff-only origin main
git log -1 --oneline
```

Cópias do mesmo sujeito/sessão/bloco só são unificadas após SHA-256 idêntico.
O caminho em ordem lexicográfica é mantido como representante **após** essa prova;
nenhum arquivo é apagado. As colunas `duplicate_paths` (lista JSON de caminhos
absolutos), `duplicate_sha256` e `duplicate_entries` registram as cópias e linhas
repetidas em `inspect` e no `inventory.csv` dentro de `tables.zip` do relatório.
Conteúdos diferentes interrompem a operação com os caminhos, hashes e identificação
do bloco. Conteúdo idêntico em sujeitos/sessões/blocos diferentes não é unificado.
Repetições do mesmo caminho com os mesmos metadados são aceitas; metadados
conflitantes são rejeitados.

No Linux, com o ambiente `bci` ativado, execute na pasta do projeto:

```bash
python run_analysis.py run --data "/home/seidi/Área de trabalho/Dados Seidi" --eda-only --output results/eda
python run_analysis.py run --data "/home/seidi/Área de trabalho/Dados Seidi" --output results/full
```

Os destinos separados preservam o EDA ao executar a análise completa. Se um destino
já existir, escolha outro ou use `--overwrite` para substituir seus arquivos gerados.

Opcionalmente, para definir grupos, edite `group` no `inventory.csv` e use-o como manifest:

```bash
python run_analysis.py run --data /caminho/para/dados --manifest results/inventory.csv
```

O manifest requer `path,subject,session,block`; `group` é opcional e fica
`unspecified` quando ausente. `path` é relativo à raiz `--data`, ou absoluto.
Um sujeito só pode pertencer a um grupo. A pasta do grupo **não** é interpretada
automaticamente como diagnóstico/condição. Arquivos fora da convenção exigem
manifest ou `input.filename_regex` com grupos `subject`, `session`, `block`.

## 3. Conferir os poucos parâmetros que dependem da aquisição

Edite `config.yaml` antes de `run`:

1. **`events.attention`**: preencher com o código real identificado no cenário
   OpenViBE ou na sequência de `markers.csv` (padrão confirmado: `786`, cruz de
   fixação/atenção). Sem ele a análise para, pois não
   existe baseline confiável. Não basta haver um código chamado `BaselineStart`:
   é preciso confirmar que corresponde à atenção deste protocolo.
2. **`events.left` / `right`**: por padrão, GDF_Left `769` / GDF_Right `770`.
   A classe da tentativa vem **exclusivamente da cue**. O marcador motor não dá a
   identidade. Label_01/Label_02, se usados no cenário, precisam ser mapeados
   explicitamente; a associação esquerda/direita não é adivinhada.
3. **`events.task_start` / `rest_start` / `trial_stop`**: preencher/conferir
   conforme o cenário. `task_start` vazio não inventa início motor. O anchor
   padrão atual é `task_start`, com `781` como início motor; `800` marca fim e pausa.
   `32775/32776` pertencem à baseline geral, não à baseline individual por atenção.
4. **`input.eeg_unit`**: começa como `native`, porque o cabeçalho fornecido não
   confirma a escala exportada pelo V-Amp/OpenViBE. Nesse modo, amplitudes/PSD
   estão em **u.a.**, e limiares físicos em µV ficam desativados; apenas canais
   exatamente constantes são detectados como planos. Após confirmar o ganho da
   aquisição/exportação, definir `uV`, `mV` ou `V`. Não inferir pela amplitude/DC.
   O ERD relativo não depende de um fator global de escala comum, mas unidades e
   ganhos continuam necessários para interpretar potência absoluta entre sujeitos.
5. **`input.channel_names`**: substituir a lista artificial pela sequência real
   dos 16 eletrodos. Depois disso, definir `figures.montage_confirmed: true`.
   A lista provisória NÃO é a montagem do EG102. Os mapas e os rótulos C3/C4 ficam
   marcados como ilustrativos enquanto não confirmados. `Aux 1/2` ficam excluídos.

Alguns códigos padrão que ajudam a ler o inventário:

| Código | Significado padrão |
|---|---|
| 32769 / 0x8001 | ExperimentStart — não é atenção |
| 32773 / 0x8005 | TrialStart |
| 32775 / 0x8007 | BaselineStart |
| 769 / 0x0301 | GDF_Left |
| 770 / 0x0302 | GDF_Right |
| 32777 / 0x8009 | RestStart |
| 800 / 0x0320 | GDF_End_Of_Trial |

São nomes padrão, não prova de como o cenário experimental foi programado.
`input.event_time: absolute` é o padrão para Event Date do CSV OpenViBE;
`row` e `relative_to_row` existem apenas para exportações com essas convenções.

## 4. Executar

```bash
python run_analysis.py run --data /caminho/para/dados --subject EG102
# Só EDA, sem treinar modelos:
python run_analysis.py run --data /caminho/para/dados --subject EG102 --eda-only
```

Sem `--subject`, processa todos os sujeitos encontrados, um de cada vez.
Caminhos no YAML são relativos ao próprio YAML; caminhos da linha de comando
são relativos ao diretório de execução. Parâmetros de linha de comando têm prioridade.

### Diagnosticar arquivos vazios ou tentativas excluídas

`run` mostra a contagem por motivo de exclusão no terminal, inclusive quando
nenhuma tentativa online é aceita. Não substitui marcadores nem encurta as janelas
para aceitar dados. Um erro de CSV vazio inclui o caminho completo e interrompe
a análise, sem publicar relatório parcial como concluído.

Para continuar explicitamente apesar de CSVs vazios, use `run --skip-empty`.
Só CSVs sem cabeçalho ou sem amostras são omitidos; outros erros continuam fatais.
Cada omissão aparece no terminal/HTML e no inventário com `status=skipped_empty`,
caminho, hash e motivo. `run-info.json` marca `partial: true`. Isso não recupera
os dados: reexporte os arquivos para uma análise completa. IM1 vazio impede o
protocolo adaptativo; um online vazio preserva a regra de parada na primeira lacuna.

Uma gravação pode ter amostras, cues e início motor, mas não conter o código de
atenção configurado. O erro inclui caminho, códigos esperados e contagens dos
códigos presentes. Cue, início motor e baseline geral não substituem a atenção
individual; confira a exportação e o protocolo antes de mudar `events.attention`.
Para continuar explicitamente sem esses blocos, acrescente
`--skip-missing-attention`: o relatório será parcial, com inventário
`status=skipped_missing_attention`, hash, motivo e códigos presentes. A contagem
de tentativas fica desconhecida (vazia), não é estimada a partir das cues.
Os marcadores originais são preservados em `events.csv` sem associação inventada
a tentativas. Essa opção é separada de `--skip-empty` e não omite outros erros.
Blocos omitidos não entram no treino; ausência de IM1 ou de um online mantém as
mesmas restrições do protocolo adaptativo.

Erros de amostras agora distinguem EEG insuficiente (menos de três amostras),
NaN, Inf, timestamps não finitos e timestamps não crescentes. O diagnóstico
inclui caminho, contagens por canal e até dez índices de linhas de dados afetadas
(base zero, não números físicos de linhas do CSV). Não são impressos valores EEG.
Para uma execução parcial que também omita esses arquivos, acrescente
`--skip-invalid-eeg`. O inventário registra `skipped_invalid_eeg`, hash, motivo e
`validation_details`; o número de tentativas permanece desconhecido. Não há
interpolação nem remoção pontual de amostras para fazer o arquivo passar.
Erros de configuração, seleção de canais, eventos ou classificadores continuam
fatais. Uma taxa incompatível/descontinuidade continua exigindo investigação;
essa opção não é um capturador genérico de qualquer erro.

As opções `--skip-empty`, `--skip-missing-attention` e `--skip-invalid-eeg`
são independentes. Omissão de um IM posterior à sessão 1 impede o comparador
congelado daquela sessão; o adaptativo continua usando apenas IM1 e online
anteriores válidos, respeitando a regra de parada em lacunas online.

Exemplo de EDA com os arquivos utilizáveis e janela fixa adequada às tarefas curtas:

```bash
python run_analysis.py run --data "/home/seidi/Área de trabalho/Dados Seidi" --eda-only --activity-window 0.5 3.5 --skip-empty --output results/eda-revisado
```

Para examinar todos os blocos mesmo se houver arquivos inválidos, sem gerar
figuras ou treinar classificadores:

```bash
python run_analysis.py diagnose --data "/home/seidi/Área de trabalho/Dados Seidi" --subject AL106 --output results/diagnostico
```

O comando salva um único `diagnostics.json` com erros por caminho, contagens de
marcadores, motivos de exclusão e tempos/status de cada tentativa (sem amostras
EEG). Se encontrar erros de leitura, conclui a inspeção e retorna código de erro
após salvar o diagnóstico. Use outro destino ou `--overwrite` para repetir.
O arquivo contém identificações dos sujeitos: mantenha-o local e compartilhe
somente os trechos necessários, anonimizados.

`missing_task_start` indica ausência do marcador motor configurado dentro da
tentativa; `short_attention_baseline`, atenção pré-cue insuficiente;
`activity_overlaps_rest_or_next_trial`, janela EDA que invade pausa/próxima tentativa;
`classification_overlaps_rest_or_next_trial`, janela de classificação que invade esse limite.
Esses motivos exigem conferir o protocolo e os marcadores reais. `0/N` sozinho
não identifica a causa. CSV vazio/sem cabeçalho exige conferir ou refazer a
exportação original; não é interpretado como um bloco sem tentativas.
Se optar por uma execução parcial sem um arquivo sabidamente vazio, exclua-o
explicitamente com `--exclude 'AL106/motor_me_AL106_3.csv'`. O bloco ficará
ausente no inventário do relatório, e a exclusão constará da configuração usada.

Cada sujeito recebe somente cinco arquivos em `results/EG102/`:

| Arquivo | Conteúdo |
|---|---|
| `report.html` | EDA comentado e figuras incorporadas; abrir no navegador local |
| `figures.pdf` | Figuras multipágina, texto/linhas vetoriais; mapas TFR rasterizados |
| `tables.zip` | Sete CSVs: inventory, events, trials, channel_qc, bandpower, scores, predictions |
| `config-used.yaml` | Parâmetros efetivamente usados |
| `run-info.json` | Versões e identificação da execução |

Hashes SHA-256 dos arquivos analisados estão no inventário. Uma falha não publica
um relatório parcial como se estivesse concluído. Para repetir, use `--overwrite`,
que substitui esses arquivos gerados e preserva outros arquivos da pasta.

Por padrão há figuras detalhadas do primeiro `me`, primeiro `im` e primeiro
online disponível, evitando centenas de páginas repetidas. **As medidas e os
classificadores usam todos os blocos disponíveis.** Para figuras de todos os
blocos, definir `figures.selection: all`.

## 5. O que o EDA calcula

O caminho de leitura do código é `eeg_io.py` → `eeg_processing.py` →
`eeg_figures.py`; `run_analysis.py` apenas organiza a execução.

- EEG bruto e depois do bandpass de 1–40 Hz, notch opcional e referência média
  opcional. Mesmos canais, janela e escala; o bruto é centrado apenas para exibição.
  O eixo usa o início da execução/imagética da tentativa de referência como zero.
  Sem esse marcador, mostra o tempo absoluto no arquivo com aviso explícito.
- Uma **janela de referência dos marcadores**, com fundos coloridos para
  atenção/cue/tarefa/pausa e linhas verticais nas marcações, sem bolinhas. Uma
  tabela dentro da imagem mostra tempos com dez casas decimais, tanto absolutos
  quanto relativos ao **início da execução/imagética (t = 0)**. Atenção, baseline
  e cue anteriores ficam em tempos negativos. A classe é escrita na cue. Quando
  não há início motor marcado, a referência não é desenhada; a figura solicita
  configurar `events.task_start`, sem usar atenção ou cue como substituto do zero.
  Essa origem visual não modifica os anchors das janelas de análise nem a baseline
  de atenção. `events.csv` preserva tempos relativos à atenção, à cue e ao início motor.
- PSD por classe e canal de interesse, comparando atenção e atividade; espectrogramas
  absolutos e relativos à baseline; curvas ERD/ERS de mu (8–13 Hz) e beta (13–30 Hz).
  C3, Cz e C4 são exibidos por padrão, ainda com nomes provisórios. A escala PSD
  usa só as frequências visíveis e é compartilhada entre classes de cada canal,
  evitando que DC e frequências fora da faixa achatem as curvas.
  Cada espectrograma vem seguido das curvas mu e beta do mesmo canal/classe,
  com eixos temporais alinhados e ICs bootstrap descritivos entre tentativas.
- Topografia da potência e do ERD no **escalpo**. Isso não é uma reconstrução de
  fontes corticais; posições artificiais não sustentam conclusões anatômicas.
  A interpolação é linear e recortada no casco convexo exato dos eletrodos
  projetados: nenhuma cor é mostrada fora da região amostrada.

Nos mapas ERD/ERS, cor forte destaca clusters pós-cue significativos; áreas
não significativas ficam atenuadas sem alterar os valores exportados. Inspirado
no [exemplo oficial do MNE](https://mne.tools/stable/auto_examples/time_frequency/time_frequency_erds.html),
o teste usa permutações de sinal por **tentativa**, bilateralmente, na diferença
de log-potência para sua baseline. Clusters agrupam bins adjacentes em tempo e
frequência, com limiar inicial t bilateral de 5% e máximo de massa por permutação.
Os p de cluster recebem ainda Bonferroni entre canais × classes do mesmo bloco.
Não há correção conjunta entre diferentes blocos ou sujeitos.

Padrões em `figures`: `tfr_significance: true`, `tfr_permutations: 512`,
`tfr_min_trials: 8` por classe e `tfr_alpha: 0.05`. Só bins pós-cue finitos em todas
as tentativas daquele painel são testados; bordas faltantes e variância zero são
excluídas. Dados insuficientes/desabilitação são indicados como **não testados**.
Método, número de tentativas, permutações efetivas e p corrigidos ficam em
`run-info.json`; parâmetros e semente ficam em `config-used.yaml`.

É uma análise exploratória dentro do sujeito, dependente da hipótese de simetria
sob troca de sinais e da permutabilidade das tentativas. Dependência temporal
entre tentativas pode violar essa hipótese. A máscara não demonstra diferenças
entre classes/grupos, não identifica precisamente início/frequência do efeito e
não equivale a provar ausência de efeito onde não há significância. Os ICs das
curvas continuam descritivos, não são a máscara inferencial.

Referência visual com **dados sintéticos**, também disponível em `marker_reference.png`:

![Sequência sintética: atenção e cue negativas, início motor em zero](marker_reference.png)

A baseline de **cada tentativa** é:

```text
[atenção + 0,2 s, cue − 0,2 s]
```

O mínimo padrão é 1 s; ajuste as margens/janelas se o protocolo tiver atenção
mais curta. Não encurtar automaticamente baselines ruins. Atividade: 0,5–4,5 s
após o anchor; classificação: **0,5–2,5 s**, uma janela única de **2 s**. As janelas
não podem invadir a pausa ou a próxima tentativa. Se forem curtas/inválidas, o
motivo fica em `trials.csv`. `status/reason` se referem à classificação;
`eda_status/eda_reason`, à atividade usada no EDA. A janela EDA inválida não exclui
uma janela de classificação válida, e vice-versa. A rejeição por artefato verifica
baseline + janela de cada finalidade separadamente, sem usar a atividade fora dos
2 s para rejeitar a classificação. `n_accepted` no inventário conta classificação;
`n_eda_accepted` conta EDA. Espectros/ERD usam somente tentativas aceitas para EDA;
treinos/predições usam somente as aceitas para classificação.

Os tempos reais fornecidos mostram tarefas de cerca de 3,75 s em blocos online e
em alguns offline: a classificação de 0,5–2,5 s cabe, mas o EDA padrão de 0,5–4,5 s
não. Para escolher explicitamente uma janela EDA fixa de 3 s, comum aos blocos:

```bash
python run_analysis.py run --data "/home/seidi/Área de trabalho/Dados Seidi" --eda-only --activity-window 0.5 3.5 --output results/eda-3s
```

`--activity-window` sobrescreve apenas `epochs.activity_s`; o valor efetivo é salvo
no relatório. Não há truncamento automático, e tentativas ainda mais curtas são
rejeitadas para EDA. A classificação permanece exatamente 2 s. Não compare
potências agregadas de relatórios com janelas diferentes como se fossem a mesma medida.

Não há ICA, correção de EOG ou detecção automática de
canais ruins; liste `bad_channels` antes de refazer a referência.

Os filtros são IIR **causais e contínuos**, sem usar amostras futuras; introduzem
atraso de fase e têm transitório inicial descartado. Seus parâmetros são fixos,
não ajustados usando o teste. A classificação recebe um bandpass adicional
8–30 Hz. Baseline é referência espectral; não se subtrai potência em µV do sinal.

Para cada tentativa, canal e banda:

```text
ERD/ERS (%) = 100 × (potência na atividade / potência na atenção − 1)
```

Negativo indica ERD; positivo, ERS. Integramos os bins `[inferior, superior)` antes
da razão de banda; não contamos 13 Hz nas duas bandas. O bootstrap reamostra
**tentativas**, não canais, com semente fixa. Seus intervalos são descritivos
dentro do sujeito, sem tratar tentativas como sujeitos independentes. TFR usa
janelas de 1 s com passo próximo de 0,1 s; a resolução verdadeira de frequência
é aproximadamente 1 Hz. Um bin temporal representa uma janela, não um instante
sem suavização. Em TFR, a baseline usa janelas totalmente contidas na atenção.

## 6. Classificação e cronologia

Implementação em `eeg_classification.py`, usando scikit-learn e pyRiemann:

| Representação | Classificadores |
|---|---|
| Ledoit–Wolf → TGSP (espaço tangente riemanniano) → StandardScaler | LDA, SVM linear, SVM RBF |
| Ledoit–Wolf → CSP com log-potência → StandardScaler | LDA, SVM linear, SVM RBF |

LDA usa shrinkage; SVM usa `C=1`, `gamma=scale`, pesos balanceados. Há até quatro
componentes CSP. Cada fold aprende sua própria representação e escala. Não há
busca de hiperparâmetros nem escolha do melhor modelo a partir do teste online.
Uma futura busca deve usar CV **aninhada** apenas no treino.

1. **`offline_cv`**: CV separada para cada `me`/`im`, até cinco folds estratificados,
   reduzidos ao número disponível de tentativas da classe minoritária. Uma
   tentativa não aparece em dois lados de um fold. Os resultados `fold=all` são
   predições fora do fold (OOF), sem duplicar as tentativas ao combinar folds.
   A CV estratificada não mede extrapolação temporal e pode ser otimista sob
   deriva; `cv_scheme: forward` ativa TimeSeriesSplit e registra sua cobertura.
2. **`session_im_frozen`**: os seis pipelines treinados no `im` da mesma sessão,
   avaliados separadamente em a, b, c, d sem atualização. É comparador diagnóstico;
   usar `im2` aqui NÃO altera o protocolo adaptativo abaixo.
3. **`adaptive_protocol`**: TGSP+SVM inicializa **somente com IM da sessão 1**.
   O mesmo modelo prediz 1a e 1b, sem usar rótulos de nenhum deles no par.
   Depois incorpora ambos e refaz cumulativamente o treino antes de 1c/1d;
   incorpora c/d antes de 2a/2b e assim por diante até 6d.
   `adaptive_update_blocks: 2` controla a cadência (1 reproduz a atualização antiga
   por bloco). **IM2–IM6 nunca entram no treino desse experimento.**

A SVM online padrão é RBF, configurável para linear. O teste ocorre antes da
atualização; não há auto-rotulação nem teste usando treino do próprio bloco.
Acurácia, acurácia balanceada, kappa, IC de Wilson da acurácia e predições por
tentativa são exportados. Os ICs binomiais são descritivos e não corrigem
dependência temporal. O campo `train_blocks` permite auditar cada treino.
As matrizes de confusão comparam o mesmo TGSP+SVM entre os protocolos, sem somar
predições de classificadores diferentes. `--eda-only` não calcula essas métricas;
o relatório sinaliza isso no início. Para acurácias e comparações, remova a opção.

Ausência de IM1 impede reproduzir o protocolo. Uma lacuna na sequência online
interrompe suas avaliações seguintes, preservando avaliações anteriores. Só
`allow_protocol_gaps: true` autoriza usar a sequência parcial observada, que é
identificada como tal no relatório. Resultados `skipped` não são acurácia zero.
Os quatro arquivos de referência incluem IM2, IM4, online1a/1b; sozinhos **não
permitem inicializar o protocolo** nem o modelo congelado da sessão 1. Para isso
é necessário pelo menos `motor_mi_EG102_1.csv`.

Esta é uma simulação supervisionada offline da atualização de modelos, compatível
com uma pseudo-coadaptação. Não reconstitui decisões/feedback do modelo original
nem demonstra adaptação do participante. Sem marcadores do feedback ou rótulos
de intenção verificados, não inferir esses efeitos a partir da acurácia.

## 7. Futuras comparações de grupos

Todas as tabelas preservam `subject,group,session,block`; potência e predições
preservam `trial_id`. `bandpower.csv` tem uma linha por tentativa × canal × banda,
com potência basal, ativa, ERD e unidade. `channel_qc.csv` preserva também a
coluna original e o estado de confirmação das posições.

Isso permite agregar primeiro por sujeito/sessão e depois comparar grupos, ou
usar modelos hierárquicos com sujeito como unidade. Não há pooling de tentativas
entre sujeitos, teste de hipótese entre grupos ou seleção de eletrodos baseada
no resultado nesta etapa. Antes disso: montagem real, unidades/ganhos,
comparabilidade de aquisição, perdas/artefatos por grupo e controle de múltiplas
comparações precisam estar definidos.

Para ler uma tabela sem extrair vários arquivos:

```python
from zipfile import ZipFile
import pandas as pd
with ZipFile("results/EG102/tables.zip") as z:
    power = pd.read_csv(z.open("bandpower.csv"))
```

## 8. Testar sem os dados reais

```bash
python -m pytest -q
python run_analysis.py demo --output demo
python run_analysis.py inspect --config demo/config.yaml
python run_analysis.py run --config demo/config.yaml
```

O gerador cria duas sessões completas de um sujeito chamado `SYNTHETIC` com
classes e marcas conhecidas, auxiliares e offsets DC. Relatórios e figuras dizem
explicitamente que são sintéticos. Não são evidência eletrofisiológica do EG102.
Os testes verificam parsing, baseline por atenção, janelas de 2 s, causalidade do
filtro, ERD conhecido, os seis pipelines e a ordem temporal/ausências do protocolo.
