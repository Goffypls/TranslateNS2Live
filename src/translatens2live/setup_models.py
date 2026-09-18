"""Descarga e instala los paquetes de argos-translate necesarios (JA->EN, EN->ES).

Uso: python -m translatens2live.setup_models
"""

from __future__ import annotations


def main() -> None:
    import argostranslate.package as argos_package

    print("Actualizando índice de paquetes de argos-translate...")
    argos_package.update_package_index()
    available = argos_package.get_available_packages()

    wanted = [("ja", "en"), ("en", "es")]
    for from_code, to_code in wanted:
        match = next(
            (p for p in available if p.from_code == from_code and p.to_code == to_code),
            None,
        )
        if match is None:
            print(f"  ! No se encontró paquete {from_code}->{to_code}, se omite.")
            continue
        print(f"  Instalando {from_code}->{to_code}...")
        path = match.download()
        argos_package.install_from_path(path)

    print("Listo.")


if __name__ == "__main__":
    main()
