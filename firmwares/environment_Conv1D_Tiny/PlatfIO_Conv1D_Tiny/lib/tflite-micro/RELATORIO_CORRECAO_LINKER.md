# Correção dos erros de vinculação do MicroMutableOpResolver

## Causa

A versão otimizada anterior compilava somente os kernels encontrados nos grafos dos três modelos. O firmware Conv1D Tiny, porém, registra um conjunto maior de operadores no `MicroMutableOpResolver<48>`. O linker precisa encontrar todas as funções `Register_*()` chamadas pelo firmware, mesmo quando o modelo carregado não utiliza esses operadores.

## Fontes restauradas

- `activations.cc` e `activations_common.cc`
- `pooling.cc` e `pooling_common.cc`
- `squeeze.cc`
- `pack.cc`
- `shape.cc`
- `pad.cc` e `pad_common.cc`
- `add.cc` e `add_common.cc`
- `sub.cc` e `sub_common.cc`
- `mul.cc` e `mul_common.cc`

Essas fontes fornecem:

- `Register_RELU()`
- `Register_RELU6()`
- `Register_AVERAGE_POOL_2D()`
- `Register_MAX_POOL_2D()`
- `Register_SQUEEZE()`
- `Register_PACK()`
- `Register_SHAPE()`
- `Register_PAD()`
- `Register_PADV2()`
- `Register_ADD()`
- `Register_SUB()`
- `Register_MUL()`

## Validações executadas

1. Compilação isolada das 76 fontes selecionadas pelo `library.json`.
2. Vinculação de um programa que referencia os 22 operadores registrados no `inference.cpp`.
3. Execução de `AllocateTensors()` e `Invoke()` nos modelos MLP, Conv1D Tiny e LSTM disponíveis.

Todos os testes concluíram sem erro.

## Após substituir a biblioteca

Limpe os objetos antigos do PlatformIO:

```powershell
pio run -t clean
pio run
```

Caso necessário, apague a pasta `.pio` do firmware e compile novamente.
