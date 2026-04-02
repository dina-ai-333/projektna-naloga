from Obdelava_podatkov import sestavi_podatke, prikazi_signal
from SPO_Komunikacijski_protokol import decode_bin_file

def main():
    paketi = decode_bin_file("LOG011.BIN")

    Fvz, signal = sestavi_podatke(paketi)

    print("Fvz:", Fvz)

    prikazi_signal(signal, f"TOF signal (Fvz={Fvz:.2f} Hz)")
    prikazi_signal(signal, "Segment", 100, 300)

if __name__ == "__main__":
    main()