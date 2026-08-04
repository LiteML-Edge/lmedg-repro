# Biblioteca TFLite Micro reduzida — LiteML

Esta cópia foi reduzida para os três firmwares do projeto:

- MLP;
- Conv1D Tiny;
- LSTM.

Ela preserva o caminho portátil para ESP32 e STM32 e também mantém, sem ativá-los por padrão, os adaptadores específicos de Xtensa e CMSIS-NN.

## Correção desta revisão

A revisão anterior considerava apenas os operadores encontrados nos FlatBuffers. Entretanto, o `inference.cpp` atual registra um conjunto maior de operadores no `MicroMutableOpResolver`. Mesmo que alguns deles não apareçam no grafo do modelo carregado, suas funções `Register_*()` precisam existir no momento da vinculação.

Foram restauradas as seguintes implementações genéricas:

- `activations.cc` + `activations_common.cc`: `RELU` e `RELU6`;
- `pooling.cc` + `pooling_common.cc`: `AVERAGE_POOL_2D` e `MAX_POOL_2D`;
- `squeeze.cc`: `SQUEEZE`;
- `pack.cc`: `PACK`;
- `shape.cc`: `SHAPE`;
- `pad.cc` + `pad_common.cc`: `PAD` e `PADV2`;
- `add.cc` + `add_common.cc`: `ADD`;
- `sub.cc` + `sub_common.cc`: `SUB`;
- `mul.cc` + `mul_common.cc`: `MUL`.

Assim, a biblioteca corresponde ao resolver atualmente usado nos firmwares, sem exigir alteração do `inference.cpp`.

## Operadores suportados pelo resolver atual

1. `UNIDIRECTIONAL_SEQUENCE_LSTM`
2. `CONV_2D`
3. `DEPTHWISE_CONV_2D`
4. `FULLY_CONNECTED`
5. `RELU`
6. `RELU6`
7. `QUANTIZE`
8. `DEQUANTIZE`
9. `MEAN`
10. `AVERAGE_POOL_2D`
11. `MAX_POOL_2D`
12. `RESHAPE`
13. `SQUEEZE`
14. `PACK`
15. `EXPAND_DIMS`
16. `SHAPE`
17. `STRIDED_SLICE`
18. `PAD`
19. `PADV2`
20. `ADD`
21. `SUB`
22. `MUL`

## Suporte preservado para ESP32

Os arquivos em `tensorflow/lite/micro/kernels/xtensa/` permanecem no pacote. Eles não são compilados pelo manifesto padrão, evitando conflito com os kernels genéricos. O caminho padrão continua adequado ao ESP32-WROOM-32/Lolin32.

## Suporte preservado para STM32

Foram mantidos:

- `tensorflow/lite/micro/cortex_m_generic/`;
- `tensorflow/lite/micro/kernels/cmsis_nn/`;
- `tensorflow/lite/micro/bluepill/`.

Os adaptadores CMSIS-NN não são compilados por padrão. Para ativá-los no STM32, o projeto deve fornecer CMSIS-Core e CMSIS-NN, incluindo `arm_nnfunctions.h`. Uma implementação CMSIS-NN deve substituir, e não ser compilada junto com, a implementação genérica equivalente.

## Dependências mantidas

- FlatBuffers;
- gemmlowp;
- ruy;
- schema TFLite;
- alocador e planejadores de memória TFLM;
- cabeçalhos exigidos pelo `MicroMutableOpResolver`;
- cabeçalhos mínimos de `signal` incluídos indiretamente pelo resolvedor.

## Uso no PlatformIO

Depois de substituir a pasta `lib`, limpe a compilação anterior antes de recompilar:

```powershell
pio run -t clean
pio run
```

Também é possível apagar manualmente a pasta `.pio` do firmware.

## Conteúdo removido

Foram removidos apenas componentes dispensáveis para estes firmwares:

- histórico Git e CI;
- exemplos, benchmarks e testes;
- dados de teste;
- ferramentas Python e codegen;
- documentação interna de desenvolvimento;
- arquiteturas ARC, CEVA, Ethos-U, Hexagon e RISC-V;
- kernels genéricos que não são usados nem registrados pelos firmwares atuais.
