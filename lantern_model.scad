// Ottoman/Turkish Hanging Lantern — 4-sided, 2-tier, pagoda-style with flared eaves
// White glass faces with dark metal mullion grids

$fn = 50;

// ─── Parameters ───

frame = 2.5;

// Two tiers
top_w = 68;  top_d = 48;  top_h = 30;   // wider
bot_w = 42;  bot_d = 32;  bot_h = 26;   // smaller, tucked underneath

// Eaves (the "flaps")
eave_overhang = 7;
eave_thick = 2.5;

// Pagoda cap
cap_h = 16;

chain_h = 12;
bracket_reach = 80;

// ─── Glass face with mullion grid ───

module glass_face(w, h, cols, rows) {
    color([0.92, 0.9, 0.85])
        cube([w, 1.2, h], center=true);

    color([0.12, 0.1, 0.08]) {
        translate([0, 0, h/2])  cube([w + 1, frame, frame], center=true);
        translate([0, 0, -h/2]) cube([w + 1, frame, frame], center=true);
        translate([w/2, 0, 0])  cube([frame, frame, h + 1], center=true);
        translate([-w/2, 0, 0]) cube([frame, frame, h + 1], center=true);

        for (c = [1:cols-1]) {
            xpos = -w/2 + c * (w / cols);
            translate([xpos, 0, 0])
                cube([frame * 0.7, frame, h], center=true);
        }
        if (rows > 1)
        for (r = [1:rows-1]) {
            zpos = -h/2 + r * (h / rows);
            translate([0, 0, zpos])
                cube([w, frame, frame * 0.7], center=true);
        }
    }
}

// ─── Tier ───

module tier(w, d, h, cols, rows) {
    gw = w - frame * 2;
    gd = d - frame * 2;
    gh = h - frame * 2;

    translate([0, d/2, 0])  glass_face(gw, gh, cols, rows);
    translate([0, -d/2, 0]) glass_face(gw, gh, cols, rows);
    translate([w/2, 0, 0])  rotate([0, 0, 90]) glass_face(gd, gh, cols, rows);
    translate([-w/2, 0, 0]) rotate([0, 0, 90]) glass_face(gd, gh, cols, rows);

    color([0.12, 0.1, 0.08])
    for (sx = [-1, 1])
        for (sy = [-1, 1])
            translate([sx * w/2, sy * d/2, 0])
                cube([frame, frame, h], center=true);

    color([0.15, 0.12, 0.1]) {
        translate([0, 0, h/2])
            cube([w + 2, d + 2, frame], center=true);
        translate([0, 0, -h/2])
            cube([w + 2, d + 2, frame], center=true);
    }
}

// ─── Flared eave (the "flap") — sloped pagoda-style overhang ───

module eave(w, d, overhang, thickness) {
    color([0.15, 0.12, 0.1])
        hull() {
            // Narrow top (attached to tier/cap)
            translate([0, 0, thickness])
                cube([w, d, 0.1], center=true);
            // Wide flared base
            translate([0, 0, 0])
                cube([w + overhang * 2, d + overhang * 2, 0.1], center=true);
        }
}

// ─── Pagoda pitched cap ───

module pagoda_cap(w, d, h) {
    color([0.15, 0.12, 0.1])
        hull() {
            cube([w, d, 0.1], center=true);
            translate([0, 0, h])
                cube([w * 0.18, d * 0.18, 0.1], center=true);
        }
}

// ─── Finial on top of cap ───

module finial() {
    color([0.12, 0.1, 0.08]) {
        cylinder(r=1.8, h=5);
        translate([0, 0, 5])
            sphere(r=2.2);
    }
}

// ─── Short chain ───

module chain() {
    color([0.18, 0.15, 0.12])
    for (i = [0:floor(chain_h/7)-1]) {
        translate([0, 0, i * 7]) {
            if (i % 2 == 0)
                difference() {
                    cube([2, 4, 7], center=true);
                    cube([0.8, 2.5, 5], center=true);
                }
            else
                difference() {
                    cube([4, 2, 7], center=true);
                    cube([2.5, 0.8, 5], center=true);
                }
        }
    }
}

// ─── Curved bracket arm ───

module bracket() {
    color([0.12, 0.1, 0.08]) {
        translate([-bracket_reach, 0, 10])
            cube([6, 30, 45], center=true);

        for (t = [0:3:90]) {
            hull() {
                translate([-bracket_reach + 6 + bracket_reach * sin(t), 0,
                           25 - bracket_reach * 0.3 * (1 - cos(t))])
                    sphere(r=2.5, $fn=10);
                translate([-bracket_reach + 6 + bracket_reach * sin(t+3), 0,
                           25 - bracket_reach * 0.3 * (1 - cos(t+3))])
                    sphere(r=2.5, $fn=10);
            }
        }

        translate([-bracket_reach + 25, 0, 15])
        rotate([90, 0, 0])
            difference() {
                cylinder(r=10, h=3, center=true, $fn=30);
                cylinder(r=7, h=4, center=true, $fn=30);
            }

        translate([5, 0, -2])
        rotate([90, 0, 0])
            difference() {
                cylinder(r=5, h=3, center=true, $fn=20);
                cylinder(r=3.5, h=4, center=true, $fn=20);
                translate([0, -5, 0])
                    cube([12, 6, 5], center=true);
            }
    }
}

// ─── Full assembly ───

module lantern() {
    // Bottom tier at origin
    z_bot = 0;
    // Mid eave (the "flap" between bottom and top tiers)
    z_mid_eave = z_bot + bot_h/2 + 1;
    // Top tier sits on the mid eave
    z_top = z_mid_eave + eave_thick + top_h/2;
    // Top eave above top tier
    z_top_eave = z_top + top_h/2;
    // Pagoda cap above top eave
    z_cap = z_top_eave + eave_thick;
    // Finial on top of cap
    z_finial = z_cap + cap_h;
    // Chain
    z_chain = z_finial + 3;
    z_bracket = z_chain + chain_h;

    // Bottom tier
    translate([0, 0, z_bot])
        tier(bot_w, bot_d, bot_h, 2, 1);

    // Mid eave (flap between tiers) — based on top tier width, hangs over bottom tier
    translate([0, 0, z_mid_eave])
        eave(top_w, top_d, eave_overhang, eave_thick);

    // Top tier
    translate([0, 0, z_top])
        tier(top_w, top_d, top_h, 3, 2);

    // Top eave above top tier (flap at base of cap)
    translate([0, 0, z_top_eave])
        eave(top_w, top_d, eave_overhang, eave_thick);

    // Pagoda cap
    translate([0, 0, z_cap])
        pagoda_cap(top_w, top_d, cap_h);

    // Finial
    translate([0, 0, z_finial])
        finial();

    // Chain
    translate([0, 0, z_chain])
        chain();

    // Bracket
    translate([0, 0, z_bracket])
        bracket();

    // Bottom cap
    color([0.15, 0.12, 0.1])
        translate([0, 0, z_bot - bot_h/2 - 2])
            cube([bot_w * 0.6, bot_d * 0.6, 3], center=true);
}

lantern();
