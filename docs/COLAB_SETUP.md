# Colab setup

Official extension: [googlecolab/colab-vscode](https://github.com/googlecolab/colab-vscode)  
User guide: [wiki/User-Guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide)  
Marketplace id: `Google.colab`

**Automatic vs manual**

- VS Code/Cursor UI: New Colab Server can pick a machine type (A100 / L4 / T4 depending on the account).
- Terminal Python in this repo: **no documented public headless allocation API**. `ColabProvider.automatic_acquisition` is false.
- `rsna runtime acquire` therefore writes a **handoff** (not a completed GPU lease). After you connect a kernel, `notebooks/02_colab_worker.ipynb` emits a handshake; the controller then runs queued jobs.
- A 300 second timeout changing the *next instruction* is not an automatic GPU swap.

OAuth uses the system browser via the extension. Do not copy cookies to undocumented endpoints.

Upload: extension “Upload to Colab” or Drive mount cell. `.env` is not uploaded.
