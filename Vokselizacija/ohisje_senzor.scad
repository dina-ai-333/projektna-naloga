// ============================================================
//  Ohisje senzorskega modula za zaznavo gest z rokami
//  (kamera + ToF senzor), gleda navzgor proti rokam
//  Projektna vaja - Uvod v racunalnisko geometrijo
//
//  SOLID model: "ploskve z debelino", votla notranjost.
//
//  STIKALI:
//    sensors_through = false -> senzorska okna so VDOLBINE
//                               -> votlina ostane ZAPRTA (vrzel)
//    sensors_through = true  -> okna prebijejo steno -> votlina ODPRTA
//    ports = true            -> kabelski port skozi zadnjo steno
//
//  Za 4. korak (demo zaznave vrzeli) pusti oboje na false.
// ============================================================

// ---------------- Glavno telo (vse v mm) ---------------------
body_l   = 60;     // sirina modula
body_w   = 30;     // globina
body_h   = 16;     // visina
wall     = 2.5;    // debelina sten  <-- vecja od velikosti voksla!
corner_r = 4;

// ---------------- Senzorska okna na zgornji ploskvi ----------
cam_r  = 5;    cam_x = -14;   // kamera
tof_r  = 4;    tof_x =  14;   // ToF senzor
led_r  = 1.5;  led_y = -10;   // LED indikator (pri x = 0)

sensors_through = false;       // glej opis zgoraj
sensor_depth = sensors_through ? (wall + 2) : 1.2;  // globina vdolbine

// ---------------- Kabelski port (zadnja -y stena) ------------
ports  = true;
usb_w  = 9;
usb_h  = 4;

// ---------------- Locljivost / st. trikotnikov ---------------
$fn = 24;          // za vokselizacijo zacni z 16-24
eps = 0.05;

// ------------------------------------------------------------
//  Pomozni modul: zaobljen kvader (hull 8 krogel)
// ------------------------------------------------------------
module rounded_box(l, w, h, r) {
    hull()
        for (sx = [-1, 1], sy = [-1, 1], sz = [-1, 1])
            translate([sx*(l/2 - r), sy*(w/2 - r), sz*(h/2 - r)])
                sphere(r);
}

// senzorsko okno: valj, ki gre z vrha navzdol za sensor_depth
module sensor_window(px, py, r) {
    translate([px, py, body_h/2 - sensor_depth/2 + eps])
        cylinder(h = sensor_depth + 2*eps, r = r, center = true);
}

// ------------------------------------------------------------
//  Glavno ohisje
// ------------------------------------------------------------
module ohisje() {
    difference() {
        // telo z votlino in senzorskimi okni
        difference() {
            rounded_box(body_l, body_w, body_h, corner_r);

            // notranja votlina
            rounded_box(body_l - 2*wall,
                        body_w - 2*wall,
                        body_h - 2*wall,
                        max(corner_r - wall, 0.6));

            // senzorska okna (vdolbine, ce sensors_through == false)
            sensor_window(cam_x, 0,     cam_r);   // kamera
            sensor_window(tof_x, 0,     tof_r);   // ToF
            sensor_window(0,     led_y, led_r);   // LED
        }

        // kabelski port skozi zadnjo (-y) steno (samo ce ports)
        if (ports)
            translate([0, -body_w/2 - 1, 0])
                cube([usb_w, 2*wall + 4, usb_h], center = true);
    }
}

ohisje();
