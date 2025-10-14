"""
Generate and rank 1000 diverse topics by interest level.

This script creates a comprehensive list of topics across various categories
and ranks them from 1-1000 based on estimated interest level (likelihood to
generate internet traffic if posted online).
"""

import csv
import random
import os
from typing import Optional, List
from utils import query_llm_for_text

def generate_topics():
    """
    Generate a diverse list of 1000 topics across multiple categories.

    Returns:
        list: List of topic strings
    """
    topics = []

    # High-interest topics (rank 1-100)
    high_interest = [
        "2024 US Presidential Election",
        "Artificial Intelligence and ChatGPT",
        "Climate Change and Global Warming",
        "Ukraine-Russia War",
        "Israel-Palestine Conflict",
        "Taylor Swift",
        "Elon Musk and Twitter/X",
        "COVID-19 Long-term Effects",
        "Cryptocurrency Crash",
        "Supreme Court Decisions 2024",
        "Mass Shootings in America",
        "TikTok Ban Debate",
        "Immigration Crisis at US Border",
        "Inflation and Economic Recession",
        "China-Taiwan Tensions",
        "Donald Trump Legal Cases",
        "Beyoncé Renaissance Tour",
        "Hollywood Strikes 2024",
        "Barbie Movie Phenomenon",
        "Oppenheimer Film",
        "NFL Season and Super Bowl",
        "LeBron James Retirement Speculation",
        "NASA Artemis Moon Mission",
        "SpaceX Starship Launch",
        "Apple Vision Pro Release",
        "iPhone 16 Launch",
        "Facebook/Meta Privacy Concerns",
        "Google Antitrust Cases",
        "Amazon Labor Disputes",
        "Student Loan Forgiveness",
        "Roe v Wade Aftermath",
        "Gun Control Legislation",
        "Electric Vehicle Adoption",
        "Tesla Cybertruck Delivery",
        "AI Art and Copyright Issues",
        "Deepfake Technology Concerns",
        "Social Media Mental Health",
        "Youth Mental Health Crisis",
        "Fentanyl Epidemic",
        "Ozempic Weight Loss Trend",
        "King Charles III Coronation",
        "Royal Family Drama",
        "World Cup 2026 Preparations",
        "Olympics 2024 Paris",
        "NBA Finals and Championships",
        "Lionel Messi MLS Move",
        "Cristiano Ronaldo Saudi League",
        "Bitcoin Price Volatility",
        "Ethereum Upgrades",
        "NFT Market Collapse",
        "Gaming Industry Layoffs",
        "Grand Theft Auto VI Announcement",
        "Marvel Cinematic Universe Phase 5",
        "Star Wars New Series",
        "Succession TV Series Finale",
        "The Last of Us HBO Adaptation",
        "Wednesday Netflix Series",
        "Stranger Things Final Season",
        "True Crime Documentaries",
        "Gabby Petito Case Legacy",
        "Idaho College Murders",
        "Missing Submarine Titanic",
        "Maui Wildfires",
        "California Earthquakes",
        "Hurricane Season 2024",
        "Tornado Outbreaks Midwest",
        "Severe Weather Climate Link",
        "Antarctic Ice Shelf Collapse",
        "Amazon Rainforest Deforestation",
        "Plastic Pollution Ocean",
        "Renewable Energy Growth",
        "Nuclear Power Renaissance",
        "Lab-Grown Meat Approval",
        "CRISPR Gene Editing Humans",
        "Alzheimer's New Treatment",
        "Cancer Breakthrough Therapies",
        "Mental Health Medication Shortage",
        "Telemedicine Expansion",
        "Health Insurance Costs",
        "Big Pharma Price Gouging",
        "Drug Price Negotiations",
        "Medicare for All Debate",
        "Universal Basic Income Trials",
        "4-Day Work Week Experiments",
        "Remote Work Policy Shifts",
        "AI Job Displacement Fears",
        "Teacher Shortages Crisis",
        "College Admissions Scandal",
        "Student Debt Crisis",
        "Book Bans in Schools",
        "Critical Race Theory Debate",
        "LGBTQ+ Rights Legislation",
        "Transgender Athletes Debate",
        "Drag Show Controversies",
        "Diversity Equity Inclusion Backlash",
        "Affirmative Action Supreme Court",
        "Police Reform Efforts",
        "Defund the Police Movement",
        "Black Lives Matter Protests"
    ]
    topics.extend(high_interest)

    # Medium-high interest (rank 101-300)
    medium_high = [
        "Housing Market Crash Fears",
        "Interest Rate Increases",
        "Bank Failures 2024",
        "Stock Market Volatility",
        "Tech Layoffs Wave",
        "Twitter Rebranding to X",
        "Instagram Algorithm Changes",
        "YouTube Monetization Updates",
        "Streaming Service Price Hikes",
        "Netflix Password Sharing Crackdown",
        "Disney+ Content Removals",
        "HBO Max Warner Bros Discovery",
        "Spotify Artist Payments",
        "Podcasting Industry Growth",
        "Joe Rogan Controversy",
        "Andrew Tate Arrest",
        "Russell Brand Allegations",
        "Harvey Weinstein Appeals",
        "Kevin Spacey Trial",
        "Bill Cosby Release",
        "R Kelly Sentencing",
        "Ghislaine Maxwell Prison",
        "Jeffrey Epstein Documents",
        "Prince Andrew Settlement",
        "Meghan Markle Lawsuits",
        "Kate Middleton Cancer Diagnosis",
        "Princess Diana Conspiracy Theories",
        "Queen Elizabeth Death Anniversary",
        "Formula 1 Drive to Survive",
        "Max Verstappen Dominance",
        "Lewis Hamilton Ferrari Move",
        "NASCAR Electric Racing",
        "IndyCar Growing Popularity",
        "Tennis Grand Slams 2024",
        "Wimbledon Championship",
        "US Open Tennis",
        "Golf LIV Merger Talks",
        "Tiger Woods Comeback Attempts",
        "Rory McIlroy Major Pursuit",
        "Boxing Jake Paul Fights",
        "UFC Fighter Pay Dispute",
        "WWE Merger with UFC",
        "Professional Wrestling Safety",
        "Esports Prize Pools Growing",
        "League of Legends Worlds",
        "Dota 2 International",
        "Counter-Strike Major Tournaments",
        "Valorant Champions Tour",
        "Fortnite World Cup",
        "Minecraft Popularity Sustained",
        "Roblox Child Safety Concerns",
        "Among Us Continued Popularity",
        "Fall Guys Game Updates",
        "Call of Duty Modern Warfare",
        "Battlefield 2042 Redemption",
        "Cyberpunk 2077 Recovery",
        "Starfield Bethesda Release",
        "Zelda Tears of Kingdom",
        "Pokemon Scarlet Violet",
        "Spider-Man Miles Morales",
        "God of War Ragnarok",
        "Horizon Forbidden West",
        "Elden Ring DLC",
        "PlayStation 5 Availability",
        "Xbox Series X Games",
        "Nintendo Switch 2 Rumors",
        "Virtual Reality Gaming Growth",
        "Augmented Reality Apps",
        "Metaverse Development Meta",
        "Web3 Technology Adoption",
        "Blockchain Practical Applications",
        "Smart Contracts Legal Status",
        "Decentralized Finance DeFi",
        "Central Bank Digital Currencies",
        "Quantum Computing Advances",
        "5G Network Expansion",
        "6G Technology Research",
        "Satellite Internet Starlink",
        "Fiber Optic Internet Expansion",
        "Net Neutrality Debates",
        "Internet Privacy Regulations",
        "Data Protection Laws GDPR",
        "Right to Be Forgotten",
        "Cookie Tracking Bans",
        "Apple Privacy Features",
        "Android Security Updates",
        "Smartphone Battery Technology",
        "Foldable Phone Improvements",
        "Wireless Charging Standards",
        "USB-C Mandates Europe",
        "E-Waste Recycling Programs",
        "Planned Obsolescence Debate",
        "Right to Repair Movement",
        "Louis Rossmann Advocacy",
        "Apple Self-Repair Program",
        "Samsung Galaxy Releases",
        "Google Pixel Phones",
        "OnePlus Flagship Devices",
        "Xiaomi Global Expansion",
        "Huawei Sanctions Impact",
        "Chinese Tech Restrictions",
        "TikTok Data Privacy",
        "WeChat Surveillance",
        "Zoom Security Concerns",
        "Microsoft Teams Growth",
        "Slack Acquisition Salesforce",
        "Discord Monetization",
        "Telegram Channel Growth",
        "Signal Private Messaging",
        "WhatsApp Encryption Debate",
        "Email Privacy Proton",
        "VPN Service Popularity",
        "Tor Browser Dark Web",
        "Cybersecurity Breaches 2024",
        "Ransomware Attacks Rising",
        "Phishing Scams Evolution",
        "Identity Theft Prevention",
        "Credit Card Fraud",
        "Online Banking Security",
        "Cryptocurrency Scams",
        "Ponzi Schemes Crypto",
        "Pyramid Schemes MLM",
        "Multi-Level Marketing Exposure",
        "Influencer Marketing Ethics",
        "Sponsored Content Disclosure",
        "Product Placement Rules",
        "Celebrity Endorsement Scandals",
        "Brand Ambassador Controversies",
        "Cancel Culture Impact",
        "Deplatforming Debates",
        "Free Speech Online",
        "Content Moderation Policies",
        "Misinformation Spreading",
        "Fact-Checking Initiatives",
        "Media Literacy Education",
        "Digital Literacy Programs",
        "Online Learning Platforms",
        "Coursera MOOCs Growth",
        "Udemy Course Quality",
        "Khan Academy Resources",
        "Duolingo Language Learning",
        "Rosetta Stone Effectiveness",
        "Study Abroad Programs",
        "Gap Year Trends",
        "Community College Value",
        "Trade School Renaissance",
        "Apprenticeship Programs",
        "Vocational Training Importance",
        "Career Changes Mid-Life",
        "Freelance Economy Growth",
        "Gig Worker Protections",
        "Uber Driver Wages",
        "Lyft Labor Disputes",
        "DoorDash Tipping Controversy",
        "Food Delivery App Fees",
        "Restaurant Industry Recovery",
        "Ghost Kitchen Trend",
        "Cloud Kitchen Business Model",
        "Meal Kit Service Popularity",
        "HelloFresh Meal Kits",
        "Blue Apron Decline",
        "Food Waste Reduction Apps",
        "Ugly Produce Movement",
        "Sustainable Agriculture",
        "Organic Farming Growth",
        "Regenerative Agriculture",
        "Permaculture Principles",
        "Urban Farming Initiatives",
        "Vertical Farming Technology",
        "Hydroponics Home Growing",
        "Aquaponics Systems",
        "Backyard Chickens Trend",
        "Beekeeping Urban Areas",
        "Composting Programs",
        "Zero Waste Lifestyle",
        "Minimalism Movement",
        "Tiny House Living",
        "Van Life Nomads",
        "RV Living Full-Time",
        "Airstream Renovation Trend",
        "Camping Popularity Surge",
        "National Parks Overcrowding",
        "Leave No Trace Ethics",
        "Outdoor Recreation Boom",
        "Hiking Trail Maintenance",
        "Mountain Biking Growth",
        "Rock Climbing Gyms",
        "Bouldering Popularity",
        "Yoga Studio Trends",
        "Pilates Reformer Classes"
    ]
    topics.extend(medium_high)

    # Medium interest (rank 301-600)
    medium = [
        "CrossFit Competition Culture",
        "Peloton Bike Sales Decline",
        "Home Gym Equipment Market",
        "Fitness Tracker Accuracy",
        "Apple Watch Health Features",
        "Fitbit Google Integration",
        "Garmin GPS Watches",
        "Sleep Tracking Technology",
        "Meditation App Subscriptions",
        "Calm App Effectiveness",
        "Headspace Meditation",
        "Therapy Apps BetterHelp",
        "Online Counseling Services",
        "Telehealth Psychiatry",
        "ADHD Medication Shortage",
        "Adderall Prescription Issues",
        "Antidepressant Withdrawal",
        "Benzodiazepine Tapering",
        "Pain Management Options",
        "Chronic Pain Support",
        "Fibromyalgia Recognition",
        "Chronic Fatigue Syndrome",
        "Long COVID Symptoms",
        "Post-Viral Syndrome",
        "Autoimmune Disease Awareness",
        "Celiac Disease Diagnosis",
        "Gluten Sensitivity Debate",
        "Food Allergies Rising",
        "Peanut Allergy Treatment",
        "EpiPen Price Controversy",
        "Insulin Cost Crisis",
        "Diabetes Type 2 Reversal",
        "Keto Diet Effectiveness",
        "Intermittent Fasting Benefits",
        "Mediterranean Diet Research",
        "Plant-Based Diet Health",
        "Vegan Nutrition Concerns",
        "Vegetarian Protein Sources",
        "Protein Powder Market",
        "Supplement Industry Regulation",
        "Vitamin D Deficiency",
        "Omega-3 Fatty Acids",
        "Probiotic Supplement Claims",
        "Gut Health Microbiome",
        "Fermented Foods Benefits",
        "Kombucha Brewing Home",
        "Sourdough Bread Making",
        "Bread Baking Pandemic",
        "Home Cooking Renaissance",
        "Meal Planning Apps",
        "Grocery Shopping Online",
        "Instacart Service Fees",
        "Amazon Fresh Delivery",
        "Walmart+ Membership",
        "Target Circle Rewards",
        "Costco Membership Worth",
        "Sam's Club Comparison",
        "Aldi Grocery Quality",
        "Trader Joe's Cult Following",
        "Whole Foods Amazon Prices",
        "Farmers Market Revival",
        "CSA Box Subscriptions",
        "Local Food Movement",
        "Farm-to-Table Restaurants",
        "Sustainable Seafood",
        "Overfishing Concerns",
        "Ocean Conservation Efforts",
        "Coral Reef Bleaching",
        "Marine Protected Areas",
        "Plastic Straw Bans",
        "Single-Use Plastic Reduction",
        "Reusable Water Bottles",
        "PFAS Forever Chemicals",
        "Water Quality Testing",
        "Flint Water Crisis Legacy",
        "Lead Pipes Replacement",
        "Clean Water Access",
        "Water Scarcity Southwest",
        "Colorado River Crisis",
        "Lake Mead Water Levels",
        "Drought California",
        "Atmospheric Rivers Weather",
        "Bomb Cyclone Events",
        "Polar Vortex Explained",
        "Heat Dome Phenomenon",
        "Urban Heat Islands",
        "Climate Refugees Migration",
        "Sea Level Rise Projections",
        "Coastal Erosion Problems",
        "Beach Replenishment Projects",
        "Hurricane Preparedness",
        "Flood Insurance Costs",
        "Wildfire Prevention Strategies",
        "Prescribed Burn Debates",
        "Forest Management Policies",
        "National Forest Logging",
        "Old Growth Forest Protection",
        "Redwood Conservation",
        "Sequoia Tree Fire Damage",
        "Biodiversity Loss Crisis",
        "Species Extinction Rates",
        "Endangered Species Act",
        "Habitat Destruction Impact",
        "Wildlife Corridor Importance",
        "Animal Migration Patterns",
        "Bird Population Decline",
        "Bee Colony Collapse",
        "Pollinator Garden Movement",
        "Native Plant Landscaping",
        "Invasive Species Problems",
        "Asian Murder Hornet",
        "Spotted Lanternfly Spread",
        "Emerald Ash Borer",
        "Dutch Elm Disease",
        "Tree Disease Management",
        "Urban Forestry Programs",
        "Million Trees Initiatives",
        "Green Infrastructure Planning",
        "Rain Garden Design",
        "Bioswale Implementation",
        "Permeable Pavement Benefits",
        "Green Roof Technology",
        "Living Walls Indoor",
        "Biophilic Design Principles",
        "Natural Light Architecture",
        "Passive Solar Design",
        "Energy Efficient Homes",
        "Net Zero Buildings",
        "LEED Certification Standards",
        "Green Building Materials",
        "Mass Timber Construction",
        "Cross-Laminated Timber",
        "Hempcrete Building Material",
        "Recycled Building Products",
        "Upcycling in Construction",
        "Circular Economy Principles",
        "Extended Producer Responsibility",
        "Product Life Cycle Assessment",
        "Carbon Footprint Calculators",
        "Personal Carbon Budgets",
        "Carbon Offset Programs",
        "Carbon Capture Technology",
        "Direct Air Capture",
        "Geoengineering Ethics",
        "Solar Radiation Management",
        "Stratospheric Aerosol Injection",
        "Cloud Seeding Weather",
        "Hurricane Modification Attempts",
        "Weather Control Conspiracy",
        "Chemtrail Theories Debunked",
        "Flat Earth Movement",
        "Moon Landing Deniers",
        "QAnon Conspiracy Theory",
        "Pizzagate Debunked",
        "Sandy Hook Denialism",
        "Holocaust Denial Illegal",
        "Historical Revisionism",
        "Misinformation Research",
        "Disinformation Campaigns",
        "Foreign Election Interference",
        "Russian Troll Farms",
        "Chinese State Media",
        "Propaganda Techniques Modern",
        "Media Bias Analysis",
        "Partisan News Sources",
        "Echo Chambers Online",
        "Filter Bubbles Social Media",
        "Algorithm Manipulation",
        "Recommendation Systems Ethics",
        "YouTube Radicalization",
        "Online Extremism",
        "White Supremacy Online",
        "Domestic Terrorism Rising",
        "January 6th Investigation",
        "Capitol Riot Consequences",
        "Insurrection Trials",
        "Seditious Conspiracy Charges",
        "Political Violence Increase",
        "Threats Against Politicians",
        "Congressional Security",
        "Supreme Court Protection",
        "Judicial Branch Independence",
        "Separation of Powers",
        "Checks and Balances Erosion",
        "Democratic Backsliding US",
        "Authoritarianism Warnings",
        "Fascism Definition Debate",
        "Political Polarization Peak",
        "Bipartisanship Decline",
        "Third Party Viability",
        "Electoral College Reform",
        "Ranked Choice Voting",
        "Gerrymandering Solutions",
        "Voting Rights Legislation",
        "Voter ID Laws",
        "Mail-In Voting Security",
        "Election Integrity Debates",
        "Vote Counting Transparency",
        "Paper Ballot Audits",
        "Electronic Voting Machines",
        "Blockchain Voting Technology",
        "Online Voting Security",
        "Same-Day Voter Registration",
        "Automatic Voter Registration",
        "Felon Voting Rights",
        "Prisoner Disenfranchisement",
        "DC Statehood Movement",
        "Puerto Rico Status Debate",
        "US Territories Representation",
        "Indigenous Rights Recognition",
        "Native American Sovereignty",
        "Tribal Land Disputes",
        "Sacred Site Protection",
        "Pipeline Protests",
        "Line 3 Pipeline",
        "Dakota Access Pipeline",
        "Keystone XL Cancellation",
        "Fossil Fuel Subsidies",
        "Oil Industry Tax Breaks",
        "Gas Price Manipulation",
        "OPEC Production Cuts",
        "Strategic Petroleum Reserve",
        "Energy Independence Goals",
        "Natural Gas Exports",
        "LNG Terminal Construction",
        "Fracking Environmental Impact",
        "Methane Leaks Problem",
        "Greenhouse Gas Emissions",
        "Paris Agreement Compliance",
        "Climate Summit COP Meetings",
        "International Climate Finance",
        "Loss and Damage Fund",
        "Climate Reparations Debate",
        "Environmental Justice Movement",
        "Cancer Alley Louisiana",
        "Industrial Pollution Hotspots",
        "Superfund Site Cleanup",
        "Brownfield Redevelopment",
        "Asbestos Exposure Risk",
        "Lead Paint Poisoning",
        "Mold Health Effects",
        "Indoor Air Quality",
        "Radon Testing Homes",
        "Carbon Monoxide Detectors",
        "Smoke Detector Batteries",
        "Fire Safety Home",
        "Electrical Fire Prevention",
        "Faulty Wiring Signs"
    ]
    topics.extend(medium)

    # Medium-low interest (rank 601-850)
    medium_low = [
        "Home Insurance Claims",
        "Property Insurance Costs",
        "Renters Insurance Worth",
        "Flood Insurance Requirement",
        "Earthquake Insurance California",
        "Homeowners Association Rules",
        "HOA Fees Rising",
        "Condo Board Politics",
        "Property Management Companies",
        "Landlord Tenant Disputes",
        "Eviction Moratorium End",
        "Rent Control Policies",
        "Affordable Housing Crisis",
        "Zoning Reform YIMBY",
        "Accessory Dwelling Units",
        "Tiny Home Regulations",
        "Mobile Home Parks",
        "Manufactured Housing Quality",
        "Modular Home Construction",
        "Prefab House Designs",
        "Container Home Living",
        "Earthship Sustainable Homes",
        "Off-Grid Living Challenges",
        "Solar Panel Installation",
        "Residential Solar Costs",
        "Solar Lease vs Buy",
        "Net Metering Policies",
        "Home Battery Storage",
        "Tesla Powerwall Price",
        "Generac Generator Demand",
        "Portable Power Stations",
        "Emergency Preparedness Kits",
        "Disaster Supply Checklist",
        "Go Bag Essentials",
        "Bug Out Bag Contents",
        "Survival Skills Training",
        "Wilderness First Aid",
        "CPR Certification Classes",
        "AED Defibrillator Access",
        "Tourniquet Use Training",
        "Stop the Bleed Campaign",
        "Blood Donation Drives",
        "Organ Donor Registration",
        "Living Donor Kidney",
        "Bone Marrow Registry",
        "Stem Cell Donation",
        "Umbilical Cord Blood Banking",
        "Genetic Testing 23andMe",
        "Ancestry DNA Results",
        "Family Tree Research",
        "Genealogy Websites",
        "Ellis Island Records",
        "Immigration History US",
        "Passenger Ship Manifests",
        "Census Data Research",
        "Historical Documents Digitization",
        "Archives Access Online",
        "Library Science Careers",
        "Librarian Degree Requirements",
        "School Library Funding",
        "Public Library Services",
        "Bookmobile Outreach",
        "Little Free Library Movement",
        "Book Donation Programs",
        "Used Bookstore Charm",
        "Independent Bookshop Support",
        "Barnes Noble Survival",
        "Amazon Books Domination",
        "Kindle E-reader Updates",
        "E-book Pricing Models",
        "Audiobook Popularity Growth",
        "Audible Subscription Worth",
        "Libby App Library",
        "OverDrive Digital Books",
        "Hoopla Streaming Library",
        "Kanopy Film Streaming",
        "Documentary Films Educational",
        "Nature Documentary Series",
        "David Attenborough Legacy",
        "Planet Earth Series",
        "Blue Planet Ocean",
        "Our Planet Netflix",
        "Nature Photography Careers",
        "Wildlife Photography Tips",
        "Bird Photography Techniques",
        "Macro Photography Insects",
        "Landscape Photography Locations",
        "Astrophotography Equipment",
        "Night Sky Photography",
        "Milky Way Photo Spots",
        "Aurora Borealis Viewing",
        "Northern Lights Iceland",
        "Southern Lights Antarctica",
        "Solar Eclipse Photography",
        "Lunar Eclipse Viewing",
        "Meteor Shower Calendar",
        "Perseid Meteor Shower",
        "Geminid Meteor Shower",
        "Leonid Meteor Shower",
        "Stargazing Best Locations",
        "Dark Sky Parks List",
        "Light Pollution Maps",
        "International Dark Sky Association",
        "Astronomy Hobby Equipment",
        "Telescope Buying Guide",
        "Binoculars Astronomy Use",
        "Star Chart Apps",
        "Planetarium Software",
        "Sky Guide App",
        "Star Walk App",
        "Night Sky App",
        "Astronomy Clubs Local",
        "Amateur Astronomy Organizations",
        "Backyard Observatory Building",
        "Astrophysics Education",
        "Space Science Degree",
        "NASA Internship Programs",
        "SpaceX Job Opportunities",
        "Blue Origin Employment",
        "Virgin Galactic Tickets",
        "Space Tourism Future",
        "Commercial Spaceflight Safety",
        "Orbital Debris Problem",
        "Space Junk Cleanup",
        "Satellite Collision Risks",
        "Kessler Syndrome Threat",
        "Starlink Satellite Constellation",
        "Internet From Space",
        "Low Earth Orbit Crowding",
        "Space Traffic Management",
        "Outer Space Treaty",
        "Space Law Development",
        "Asteroid Mining Rights",
        "Moon Property Claims",
        "Mars Colonization Plans",
        "SpaceX Mars Mission",
        "NASA Mars Samples",
        "Mars Rover Discoveries",
        "Perseverance Rover Updates",
        "Ingenuity Helicopter Mars",
        "James Webb Telescope Images",
        "Hubble Telescope Legacy",
        "Exoplanet Discoveries",
        "Habitable Zone Planets",
        "SETI Search Aliens",
        "Drake Equation Updated",
        "Fermi Paradox Explanations",
        "Wow Signal Mystery",
        "UFO UAP Government Reports",
        "Pentagon UFO Videos",
        "Navy Pilot Sightings",
        "Area 51 Secrets",
        "Roswell Incident Anniversary",
        "Ancient Aliens Theories",
        "Pyramids Construction Methods",
        "Stonehenge Purpose Debate",
        "Easter Island Moai",
        "Machu Picchu History",
        "Lost City Atlantis",
        "Underwater Archaeology",
        "Shipwreck Discoveries",
        "Titanic Wreck Expeditions",
        "Treasure Hunting Legal",
        "Metal Detecting Hobby",
        "Coin Collecting Values",
        "Stamp Collecting Philately",
        "Baseball Card Investing",
        "Trading Card Games",
        "Magic the Gathering",
        "Pokemon Card Values",
        "Yu-Gi-Oh Tournaments",
        "Board Game Renaissance",
        "Dungeons Dragons Popularity",
        "Role-Playing Games Growth",
        "Tabletop Gaming Cafes",
        "Chess Resurgence Online",
        "Magnus Carlsen Dominance",
        "Chess Prodigy Stories",
        "Chess Engine Development",
        "AlphaZero Chess AI",
        "Stockfish Chess Engine",
        "Chess Cheating Scandals",
        "Poker Online Legality",
        "Casino Gambling Laws",
        "Sports Betting Expansion",
        "FanDuel DraftKings Growth",
        "Problem Gambling Help",
        "Gambling Addiction Treatment",
        "Alcoholics Anonymous Effectiveness",
        "12-Step Program Alternatives",
        "SMART Recovery Program",
        "Harm Reduction Approach",
        "Needle Exchange Programs",
        "Safe Injection Sites",
        "Drug Decriminalization Oregon",
        "Portugal Drug Policy",
        "War on Drugs Failure",
        "Prison Reform Needed",
        "Mass Incarceration Crisis",
        "Private Prison Industry",
        "Solitary Confinement Effects",
        "Prison Labor Exploitation",
        "Recidivism Rates High",
        "Reentry Programs Ex-Offenders",
        "Ban the Box Movement",
        "Criminal Record Expungement",
        "Restorative Justice Practices",
        "Victim Offender Mediation",
        "Community Service Sentences",
        "Probation System Reform",
        "Parole Board Decisions",
        "Life Sentence Commutations",
        "Death Penalty Abolition",
        "Capital Punishment Debate",
        "Innocence Project Work",
        "Wrongful Conviction Cases",
        "DNA Exoneration Stories"
    ]
    topics.extend(medium_low)

    # Low interest (rank 851-1000)
    low_interest = [
        "Legal Aid Services",
        "Public Defender Shortage",
        "Court Appointed Attorneys",
        "Civil Legal Assistance",
        "Tenant Rights Organizations",
        "Consumer Protection Agencies",
        "Better Business Bureau",
        "Federal Trade Commission",
        "Securities Exchange Commission",
        "Financial Fraud Prevention",
        "Elder Financial Abuse",
        "Power of Attorney Documents",
        "Living Will Importance",
        "Advance Healthcare Directive",
        "Do Not Resuscitate Orders",
        "Hospice Care Options",
        "Palliative Care Benefits",
        "End-of-Life Planning",
        "Estate Planning Basics",
        "Will Writing Services",
        "Trust Fund Establishment",
        "Probate Process Length",
        "Inheritance Tax Laws",
        "Gift Tax Exclusions",
        "Charitable Donations Tax",
        "501c3 Nonprofit Status",
        "Nonprofit Fundraising Tips",
        "Grant Writing Skills",
        "Foundation Grant Programs",
        "Community Foundation Role",
        "Donor-Advised Funds",
        "Planned Giving Strategies",
        "Endowment Fund Management",
        "Scholarship Fund Creation",
        "College Savings Plans 529",
        "Education Savings Accounts",
        "FAFSA Application Help",
        "Financial Aid Packages",
        "Student Grant Programs",
        "Pell Grant Eligibility",
        "Work-Study Jobs College",
        "Graduate School Funding",
        "PhD Stipend Amounts",
        "Postdoctoral Fellowship Pay",
        "Academic Job Market Crisis",
        "Adjunct Professor Wages",
        "Tenure Track Positions",
        "Academic Publishing Models",
        "Open Access Journals",
        "Peer Review Process",
        "Research Reproducibility Crisis",
        "Scientific Method Teaching",
        "STEM Education Importance",
        "Math Anxiety Solutions",
        "Reading Comprehension Skills",
        "Literacy Programs Adults",
        "ESL Classes Community",
        "Bilingual Education Benefits",
        "Foreign Language Immersion",
        "Study Abroad Scholarships",
        "International Exchange Programs",
        "Fulbright Scholarship Info",
        "Peace Corps Service",
        "AmeriCorps Volunteer",
        "Teach For America",
        "City Year Program",
        "Habitat for Humanity",
        "Red Cross Volunteer",
        "Food Bank Volunteering",
        "Soup Kitchen Service",
        "Homeless Shelter Help",
        "Animal Shelter Volunteering",
        "Wildlife Rehabilitation Centers",
        "Marine Mammal Rescue",
        "Bird Rescue Organizations",
        "Raptor Center Programs",
        "Wildlife Hospital Donations",
        "Conservation Volunteer Work",
        "Trail Maintenance Volunteering",
        "Beach Cleanup Events",
        "River Cleanup Projects",
        "Adopt-a-Highway Programs",
        "Community Garden Plots",
        "Victory Garden Movement",
        "Seed Library Exchanges",
        "Heirloom Seed Preservation",
        "Seed Saving Techniques",
        "Plant Propagation Methods",
        "Cutting Propagation Guide",
        "Division Propagation Plants",
        "Layering Propagation Technique",
        "Grafting Fruit Trees",
        "Budding Technique Roses",
        "Pruning Best Practices",
        "Tree Trimming Safety",
        "Arborist Certification",
        "Landscape Architecture Degree",
        "Horticulture Career Paths",
        "Botanical Garden Employment",
        "Parks Department Jobs",
        "Recreation Center Programs",
        "Youth Sports Leagues",
        "Little League Baseball",
        "Youth Soccer Organization",
        "Swim Team Benefits Kids",
        "Gymnastics Training Young",
        "Dance Classes Children",
        "Ballet School Enrollment",
        "Music Lessons for Kids",
        "Piano Teacher Qualifications",
        "Guitar Lessons Beginner",
        "Violin Suzuki Method",
        "Brass Instrument Maintenance",
        "Woodwind Reed Selection",
        "Percussion Technique Practice",
        "Drum Set Buying Guide",
        "Electronic Drum Pads",
        "MIDI Controller Options",
        "Music Production Software",
        "Digital Audio Workstation",
        "Audio Interface Selection",
        "Studio Monitor Speakers",
        "Microphone Types Recording",
        "Condenser Microphone Use",
        "Dynamic Microphone Live",
        "Ribbon Microphone Vintage",
        "Acoustic Treatment Studio",
        "Soundproofing Materials",
        "Bass Trap Placement",
        "Diffuser Panel Design",
        "Recording Booth DIY",
        "Vocal Booth Portable",
        "Home Recording Setup",
        "Podcasting Equipment List",
        "USB Microphone Quality",
        "XLR Cable Length",
        "Phantom Power Requirement",
        "Audio Mixer Basics",
        "Compressor Settings Vocals",
        "EQ Frequency Chart",
        "Reverb Plugin Types",
        "Delay Effect Creative",
        "Chorus Effect Guitar",
        "Distortion Pedal Options",
        "Guitar Amp Modeling",
        "Bass Amp Head Cabinet",
        "Speaker Cabinet Wiring",
        "Ohm Rating Speakers",
        "Speaker Cabinet Wiring",
        "Soldering Iron Temperature",
        "Circuit Board Repair",
        "Capacitor Replacement Guide",
        "Resistor Color Code Chart",
        "LED Circuit Basics",
        "Arduino Project Ideas",
        "Raspberry Pi Uses",
        "3D Printer Calibration",
        "Filament Types Comparison",
        "Resin Printing Settings",
        "CNC Machine Programming",
        "Laser Cutter Safety",
        "Vinyl Cutter Projects",
        "Heat Press Temperature",
        "Screen Printing Techniques",
        "Block Printing Supplies",
        "Linocut Carving Tools",
        "Woodcut Printmaking",
        "Etching Press Types",
        "Lithography Stone Preparation",
        "Monotype Printing Process",
        "Collagraph Plate Making",
        "Relief Printing Methods",
        "Intaglio Printing Ink",
        "Silkscreen Mesh Count",
        "Stencil Cutting Tips",
        "Spray Paint Techniques",
        "Airbrush Cleaning Methods",
        "Acrylic Pouring Medium",
        "Oil Paint Drying Time",
        "Watercolor Paper Types",
        "Gouache Painting Tips",
        "Tempera Paint Uses",
        "Encaustic Wax Medium",
        "Fresco Painting Traditional",
        "Mural Painting Outdoor",
        "Trompe-l'oeil Technique",
        "Grisaille Underpainting",
        "Glazing Layers Oil",
        "Scumbling Technique Paint",
        "Impasto Texture Application",
        "Alla Prima Wet-on-Wet",
        "Chiaroscuro Lighting Effect",
        "Sfumato Blending Leonardo",
        "Color Theory Basics",
        "Complementary Colors Chart",
        "Analogous Color Schemes",
        "Triadic Color Harmony",
        "Monochromatic Palette",
        "Color Temperature Warm Cool",
        "Value Scale Grayscale",
        "Hue Saturation Brightness",
        "Color Mixing Guide",
        "Primary Colors Traditional",
        "Secondary Colors Mixing",
        "Tertiary Colors Creation",
        "Neutral Colors Gray",
        "Earth Tones Palette",
        "Pastel Colors Soft",
        "Jewel Tones Rich",
        "Neon Colors Bright",
        "Metallic Paint Effects",
        "Iridescent Pigments",
        "Fluorescent Colors UV",
        "Phosphorescent Glow Paint",
        "Thermochromic Color Change",
        "Photochromic Light Sensitive",
        "Hydrochromic Water Reactive",
        "Electrochromic Voltage Change",
        "Piezochromic Pressure Sensitive",
        "Solvatochromic Solvent Dependent",
        "Textile Dyeing Methods",
        "Natural Dye Plants",
        "Indigo Vat Dyeing",
        "Mordant Types Fabric",
        "Tie-Dye Patterns",
        "Batik Wax Resist",
        "Shibori Japanese Technique",
        "Ice Dyeing Process",
        "Low Immersion Dyeing",
        "Dip Dyeing Gradient",
        "Ombre Effect Fabric",
        "Bleach Discharge Dyeing",
        "Fabric Painting Techniques",
        "Textile Printing Methods",
        "Block Printing Fabric",
        "Stamp Making Rubber",
        "Potato Printing Kids",
        "Leaf Printing Nature",
        "Monoprinting Fabric",
        "Sun Printing Cyanotype",
        "Eco Printing Plants",
        "Rust Dyeing Metal",
        "Contact Printing Leaves",
        "Anthotype Photography Plant",
        "Alternative Process Photography",
        "Film Photography Revival",
        "Darkroom Setup Home"
    ]
    topics.extend(low_interest)

    return topics


# Prompt template used for LLM-based topic generation
TOPIC_GENERATION_PROMPT = """Generate a diverse list of exactly {n_topics} topics from current news, internet trends, and general knowledge.

Requirements:
1. Include topics across ALL these categories:
   - Politics and current events
   - Technology and AI
   - Entertainment and celebrities
   - Sports and athletics
   - Science and health
   - Environment and climate
   - Culture and society
   - Business and economics
   - Education and learning
   - Hobbies and crafts
   - Arts and media
   - Food and lifestyle
   - Travel and geography
   - History and archaeology
   - Philosophy and ideas

2. Range from highly interesting mainstream topics (that would go viral online) to niche specialized topics (that few would click on)

3. Mix of:
   - Controversial and divisive topics (high buzz potential)
   - Popular mainstream topics (high interest, low controversy)
   - Specialized professional topics (medium interest)
   - Academic and educational topics (lower interest)
   - Niche hobby and craft topics (low interest)

4. Return as a simple numbered list, one topic per line, in the format:
   1. Topic Name
   2. Topic Name
   ...

Do not include any explanations, commentary, or categories in your response - ONLY the numbered list of topics."""


def generate_topics_with_llm(
    n_topics: int = 1000,
    provider: str = "claude",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.8
) -> List[str]:
    """
    Generate diverse topics using an LLM.

    This function queries either OpenAI or Claude to generate a diverse list of topics
    spanning multiple categories and interest levels, suitable for ranking by buzz/interest.

    Args:
        n_topics: Number of topics to generate (default: 1000)
        provider: Either "openai" or "claude" (default: "claude")
        api_key: API key (if None, uses environment variables)
        model: Model to use (if None, uses default for provider)
        temperature: Sampling temperature (default: 0.8 for good variety)

    Returns:
        list: List of topic strings

    Raises:
        ValueError: If provider is invalid or API key not found

    Example:
        >>> topics = generate_topics_with_llm(n_topics=100, provider="claude")
        >>> print(f"Generated {len(topics)} topics")

    Note:
        The prompt used for generation is stored in the TOPIC_GENERATION_PROMPT constant
        at the top of this module for reference.
    """
    provider = provider.lower()

    if provider not in ["openai", "claude"]:
        raise ValueError("Provider must be either 'openai' or 'claude'")

    # Get API key from environment if not provided
    if api_key is None:
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
        else:  # claude
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key not provided and ANTHROPIC_API_KEY environment variable not set")

    # Set default model
    if model is None:
        model = "gpt-4o-mini" if provider == "openai" else "claude-sonnet-4-5-20250929"

    # Generate system and user prompts
    system_prompt = """You are a helpful assistant that generates diverse lists of topics spanning multiple categories and interest levels. You follow instructions precisely and return only the requested output format."""

    user_prompt = TOPIC_GENERATION_PROMPT.format(n_topics=n_topics)

    print(f"Querying {provider} ({model}) to generate {n_topics} topics...")

    # Query the LLM
    response_text = query_llm_for_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider=provider,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=8000  # Need enough tokens for 1000 topics
    )

    # Parse the response to extract topics
    topics = []
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # Remove numbering (e.g., "1. Topic Name" -> "Topic Name")
        # Handle various formats: "1.", "1)", "1 -", etc.
        import re
        match = re.match(r'^\d+[\.\)\-\:]?\s*(.*)', line)
        if match:
            topic = match.group(1).strip()
            if topic:
                topics.append(topic)
        else:
            # Line doesn't start with number, but might still be a topic
            if line and not line.startswith('#'):
                topics.append(line)

    print(f"Successfully extracted {len(topics)} topics from LLM response")

    return topics


def generate_topics_with_llm_openrouter(
    n_topics: int = 1000,
    model: str = "anthropic/claude-sonnet-4",
    api_key: Optional[str] = None,
    temperature: float = 0.8
) -> List[str]:
    """
    Generate diverse topics using an LLM via OpenRouter.

    Args:
        n_topics: Number of topics to generate (default: 1000)
        model: OpenRouter model identifier (default: "anthropic/claude-sonnet-4")
                Format: "provider/model-name" (e.g., "openai/gpt-4o-mini")
        api_key: OpenRouter API key (if None, uses OPENROUTER_API_KEY environment variable)
        temperature: Sampling temperature (default: 0.8 for good variety)

    Returns:
        list: List of topic strings

    Raises:
        ValueError: If API key not found

    Example:
        >>> topics = generate_topics_with_llm_openrouter(n_topics=100, model="anthropic/claude-sonnet-4")
        >>> print(f"Generated {len(topics)} topics")

    Note:
        The prompt used for generation is stored in the TOPIC_GENERATION_PROMPT constant
        at the top of this module for reference.
    """
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key not provided and OPENROUTER_API_KEY environment variable not set")

    # Generate system and user prompts
    system_prompt = """You are a helpful assistant that generates diverse lists of topics spanning multiple categories and interest levels. You follow instructions precisely and return only the requested output format."""

    user_prompt = TOPIC_GENERATION_PROMPT.format(n_topics=n_topics)

    print(f"Querying OpenRouter ({model}) to generate {n_topics} topics...")

    # Query via OpenRouter
    from utils import query_llm_for_text_openrouter
    response_text = query_llm_for_text_openrouter(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=8000  # Need enough tokens for 1000 topics
    )

    # Parse the response to extract topics
    topics = []
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # Remove numbering (e.g., "1. Topic Name" -> "Topic Name")
        # Handle various formats: "1.", "1)", "1 -", etc.
        import re
        match = re.match(r'^\d+[\.\)\-\:]?\s*(.*)', line)
        if match:
            topic = match.group(1).strip()
            if topic:
                topics.append(topic)
        else:
            # Line doesn't start with number, but might still be a topic
            if line and not line.startswith('#'):
                topics.append(line)

    print(f"Successfully extracted {len(topics)} topics from LLM response")

    return topics


def rank_topics_by_interest(topics):
    return [(rank + 1, topic) for rank, topic in enumerate(topics)]


def save_to_csv(ranked_topics, filename='topic_rankings.csv'):
    """
    Save ranked topics to CSV file.

    Args:
        ranked_topics: List of (rank, topic) tuples
        filename: Output CSV filename
    """
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'topic'])
        for rank, topic in ranked_topics:
            writer.writerow([rank, topic])

    print(f"Saved {len(ranked_topics)} ranked topics to {filename}")


def main():
    """Main execution function."""
    print("Generating 1000 diverse topics...")
    topics = generate_topics_with_llm_openrouter()

    print(f"Generated {len(topics)} topics")

    # print("Ranking topics by interest level...")
    ranked_topics = rank_topics_by_interest(topics)

    print("Saving to CSV...")
    save_to_csv(ranked_topics)


if __name__ == "__main__":
    main()
