# Bot de Xadrez com Aprendizado por Reforço

...

## Como Executar a Interface Gráfica

Para jogar contra o bot usando a GUI:

```bash
python src/gui.py

Note que, na implementação atual, os movimentos do usuário são inseridos via console. Futuramente, pode-se melhorar a GUI para aceitar cliques do mouse.


---

### **Atualização da Estrutura do Projeto**

chess_bot/ ├── README.md ├── requirements.txt ├── src/ │ ├── main.py │ ├── chess_env.py │ ├── agent.py │ ├── utils.py │ ├── gui.py └── models/ └── model.pth

---

### **Considerações Finais**

- **Desempenho do Modelo:** Devido à complexidade do xadrez, o modelo pode não apresentar desempenho competitivo sem treinamento extensivo.
- **Recursos Computacionais:** O treinamento pode ser intensivo; considere usar GPUs para acelerar o processo.
- **Melhorias Futuras:**
  - Implementar uma GUI completa com interação via mouse.
  - Aperfeiçoar a função de avaliação do tabuleiro.
  - Explorar arquiteturas de rede mais avançadas, como redes convolucionais.

---

Espero que estas atualizações completem o projeto de acordo com suas necessidades. O projeto agora possui um mapeamento de ações definido, capacidade de salvar e carregar modelos, registro de métricas e uma interface gráfica básica para interação.

Se tiver mais dúvidas ou precisar de assistência adicional, estou à disposição!