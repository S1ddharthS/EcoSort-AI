-- EcoSort AI database schema and seed data.
-- Run automatically by database/db.py on first start.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path      TEXT    NOT NULL,
    predicted_class TEXT    NOT NULL
                    CHECK (predicted_class IN
                           ('plastic', 'paper', 'cardboard', 'glass', 'metal', 'organic')),
    confidence      REAL    NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_predictions_created
    ON predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_class
    ON predictions (predicted_class);

CREATE TABLE IF NOT EXISTS recommendations (
    category             TEXT PRIMARY KEY,
    recyclable           TEXT NOT NULL,
    bin_colour           TEXT NOT NULL,
    how_to_prepare       TEXT NOT NULL,  -- JSON array of steps
    do_list              TEXT NOT NULL,  -- JSON array
    dont_list            TEXT NOT NULL,  -- JSON array
    fun_fact             TEXT NOT NULL,
    environmental_impact TEXT NOT NULL
);

-- Seed recommendations. JSON arrays are stored as text and parsed by db.py.
-- Recyclability varies by municipality, so rules that commonly differ are
-- marked "Depends on your area" instead of being presented as universal.
INSERT OR IGNORE INTO recommendations
(category, recyclable, bin_colour, how_to_prepare, do_list, dont_list, fun_fact, environmental_impact)
VALUES
(
    'plastic',
    'Depends on your area',
    'Blue recycling bin (check local rules)',
    '["Empty the container completely.", "Rinse it and let it dry.", "Keep the lid on if your area accepts it.", "Flatten bottles to save space."]',
    '["Rigid bottles and jugs", "Food tubs and trays", "Shampoo and detergent bottles", "Clean plastic lids (check the size)"]',
    '["Plastic bags and cling film", "Polystyrene foam", "Dirty food containers", "Black plastic, which sorting machines often miss"]',
    'A plastic bottle can take hundreds of years to break down in landfill. Recycling gives the material another useful life.',
    'Recycling plastic usually uses less energy than making new plastic from oil. The exact saving depends on the plastic type and how your area recycles it.'
),
(
    'paper',
    'Yes',
    'Blue recycling bin',
    '["Keep paper clean and dry.", "Flatten sheets and boxes.", "Remove plastic tape, wrap and staples.", "Place loose in the bin, not in a plastic bag."]',
    '["Newspapers and magazines", "Office paper and envelopes", "Clean paper bags", "Junk mail and catalogues"]',
    '["Greasy pizza boxes", "Paper towels and napkins", "Waxed or plastic-coated paper", "Used tissues"]',
    'Paper fibres can be recycled about five to seven times before they become too short to use.',
    'Recycling paper reduces demand for new wood pulp and generally uses less energy and water than making paper from virgin fibre.'
),
(
    'cardboard',
    'Yes',
    'Blue recycling bin',
    '["Empty out any food or packing material.", "Remove plastic tape and bubble wrap.", "Flatten the box so it fits and saves space.", "Keep it dry."]',
    '["Shipping and moving boxes", "Cereal and snack boxes", "Clean egg cartons", "Brown paper bags"]',
    '["Greasy pizza boxes", "Wet or mouldy cardboard", "Cardboard with heavy food residue", "Foam or plastic packaging"]',
    'Cardboard can be recycled several times, and each time it replaces new wood pulp.',
    'Recycling cardboard keeps a useful fibre in circulation and reduces the demand for fresh timber and the energy used to process it.'
),
(
    'glass',
    'Depends on your area',
    'Green or blue glass bank (check local rules)',
    '["Empty and rinse the bottle or jar.", "Remove non-glass lids and caps.", "Leave labels on; they are removed in recycling.", "Do not break it on purpose; intact glass is safer to handle."]',
    '["Glass bottles", "Glass jars", "Food and drink containers", "Clean glass lids (where accepted)"]',
    '["Window glass and mirrors", "Pyrex and ovenware", "Ceramics, cups and plates", "Light bulbs"]',
    'Glass can be recycled again and again without losing its quality.',
    'Recycling glass reduces the need for new sand and saves energy compared with melting raw materials from scratch.'
),
(
    'metal',
    'Yes',
    'Blue or grey recycling bin (check local rules)',
    '["Empty and rinse the can or tin.", "Remove food residue.", "Crush cans only if your area asks you to.", "Place loose in the recycling bin."]',
    '["Aluminium drink cans", "Steel food cans", "Clean foil trays, where accepted", "Metal jar lids and bottle caps"]',
    '["Batteries", "Electronics and cables", "Paint and chemical cans", "Sharp blades or mixed metal items"]',
    'Recycling aluminium saves about 95% of the energy needed to make it from raw ore.',
    'Metals are highly recyclable and keep their properties. Recycling aluminium and steel avoids the mining and energy needed to produce new metal.'
),
(
    'organic',
    'No',
    'Green or brown organics bin (or home compost)',
    '["Separate food scraps from dry recycling.", "Drain excess liquid.", "Use a compost caddy or wrap scraps in newspaper.", "Do not use plastic bags, even if they say compostable, unless your council accepts them."]',
    '["Fruit and vegetable scraps", "Coffee grounds and paper filters", "Plastic-free tea bags", "Eggshells and garden trimmings"]',
    '["Plastic and compostable bags (unless accepted locally)", "Pet waste", "Treated or painted wood", "Large amounts of cooking oil"]',
    'When food scraps rot in landfill without oxygen, they can produce methane, a potent greenhouse gas. Composting returns nutrients to the soil.',
    'Composting keeps organic waste out of landfill and creates a nutrient-rich soil amendment that helps plants grow.'
);
