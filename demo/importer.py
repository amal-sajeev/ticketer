import requests
import json

# url = "http://localhost:8000/knowledge/documentation"

# payload = [
#   {"title": "Router Reset Procedure", "content": "To reset your STC router: 1) Locate reset button on back 2) Press with paperclip for 10 seconds 3) Wait 5 minutes for reboot. Default credentials: admin/stc@123", "category": "technical"},
#   {"title": "Bill Payment Options", "content": "Pay STC bills via: 1) STC Pay app 2) Bank transfers 3) ATM 4) Authorized retailers. Late fees apply after 15th of month.", "category": "billing"},
#   {"title": "5G Coverage Areas", "content": "Current 5G coverage: Riyadh (90%), Jeddah (85%), Dammam (80%). Check coverage map in MySTC app. Requires compatible device.", "category": "internet"},
#   {"title": "SIM Replacement", "content": "Lost SIM replacement: 1) Visit STC store with ID 2) Pay 25 SAR fee 3) Activation within 2 hours. Keep old SIM blocked via app.", "category": "account"},
#   {"title": "International Roaming", "content": "Activate roaming: SMS 'ROAM' to 900. Packages: Gulf 50 SAR/week, Global 100 SAR/week. Data speed capped at 512kbps.", "category": "general"},
#   {"title": "STC Fiber Activation Process", "content": "New fiber activation takes 48-72 hours after payment. Technician visit required for ONT installation. Activation SMS will confirm completion.", "category": "technical"},
#   {"title": "Bill Dispute Process", "content": "Submit dispute via MySTC app within 30 days. Required: 1) Screenshot of issue 2) Previous bill PDF 3) Bank statement if applicable. Resolution in 5-7 working days.", "category": "billing"},
#   {"title": "5G Device Compatibility", "content": "Check IMEI on stc.com/5g. Minimum requirements: Snapdragon 855+/Dimensity 1000+ chipset, SA/NSA support. Non-compatible devices fallback to 4G.", "category": "internet"},
#   {"title": "Business Account Transfer", "content": "Required documents: 1) Commercial Registration 2) Authorized Signatory ID 3) Existing account holder approval letter. Process takes 3 business days.", "category": "account"},
#   {"title": "International Roaming Packages", "content": "GCC: 50 SAR/week (5GB). Europe: 150 SAR/week (10GB). USA: 200 SAR/week (15GB). Activate via *888# before travel.", "category": "general"},
#   {"title": "WiFi Calling Setup", "content": "Enable in Phone Settings > Cellular > WiFi Calling. Requires: 1) iOS 12+/Android 10+ 2) STC SIM 3) Enabled in MySTC account.", "category": "technical"},
#   {"title": "Late Payment Penalties", "content": "15+ days late: 2.5% of bill. 30+ days: Service suspension. 60+ days: 10 SAR reconnection fee after payment.", "category": "billing"},
#   {"title": "Fiber Outage Reporting", "content": "Check stc.com/outages first. If no reported issue: 1) Power cycle ONT 2) Check fiber cable bends 3) Report via app with ONT light photos.", "category": "internet"},
#   {"title": "SIM Ownership Transfer", "content": "Both parties must visit STC store with: 1) Original IDs 2) Current bill 3) Signed transfer form. Prepaid balance non-transferable.", "category": "account"},
#   {"title": "STC TV Channel List", "content": "Basic: 120 channels. Premium: +50 sports/movies. Ultra: +30 4K channels. Channel numbers 100-299. Package changes apply next billing cycle.", "category": "general"},
#   {"title": "Router Firewall Settings", "content": "Default admin access: 192.168.1.1. Recommended settings: 1) WPA3 encryption 2) MAC filtering 3) Disable WPS. Port forwarding requires business account.", "category": "technical"},
#   {"title": "Auto-Pay Enrollment", "content": "Link credit card in MySTC app for 5% discount. Failed payments retry after 24h. Disable auto-pay before SIM cancellation.", "category": "billing"},
#   {"title": "Mobile Hotspot Limits", "content": "Consumer plans: 50GB/month tethering. Business plans: Unlimited. Exceeding limits throttles to 1Mbps. Reset at billing cycle.", "category": "internet"},
#   {"title": "Lost Phone Protection", "content": "Immediately: 1) Call 920020540 to block SIM 2) Report to police 3) Submit report via app for replacement. 50 SAR replacement fee.", "category": "account"},
#   {"title": "STC Business Hotline", "content": "Dedicated line: 920001234. Operating hours: Sun-Thu 8AM-8PM. Priority routing for accounts with 50+ lines.", "category": "general"},
#   {"title": "IPv6 Migration Guide", "content": "New fiber connections default to IPv6. Legacy devices may need: 1) Firmware update 2) Dual-stack configuration 3) Disable IPv6 firewall temporarily.", "category": "technical"},
#   {"title": "VAT Exemption Process", "content": "Government entities submit: 1) Exemption certificate 2) Ministry letter 3) Updated account details. Refunds processed quarterly.", "category": "billing"},
#   {"title": "Gaming Package Details", "content": "Gamer Pro: 100 SAR/month (ping<20ms to EU/MENA servers). Includes: 1) Priority traffic 2) Free Discord Nitro 3) Game patches overnight.", "category": "internet"},
#   {"title": "Employee SIM Management", "content": "Admin portal allows: 1) Usage limits 2) Service restrictions 3) Department billing codes. Requires HR system integration approval.", "category": "account"},
#   {"title": "Hajj Season Promotions", "content": "Special packages: 1) Free local minutes 2) Extra 20GB data 3) Makkah/Madinah coverage boost. Auto-activates for pilgrims near holy sites.", "category": "general"}
# ]

# headers = {
#   'Content-Type': 'application/json'
# }

# #service_memory

# for i in payload: 
#     response = requests.request("POST", url, headers=headers, data=json.dumps(i))
#     print(response.text)

# url = "http://localhost:8000/knowledge/service-memory"

# payload = [
#   {"query": "Internet slow in Riyadh", "resolution": "Confirmed network congestion in Al-Nakheel area. Suggested: 1) Use 5GHz band 2) Schedule downloads after midnight 3) Temporary speed boost available", "category": "internet", "agent_name": "tech_salim"},
#   {"query": "Double billing for international calls", "resolution": "Applied 50 SAR credit for billing error on GCC calls. System glitch fixed in latest update. Advised to check next bill.", "category": "billing", "agent_name": "billing_nora"},
#   {"query": "Can't access MySTC account", "resolution": "Reset 2FA via SMS verification. Cleared cache in app. Recommended password change every 90 days.", "category": "account", "agent_name": "support_khalid"},
#   {"query": "TV channels pixelating", "resolution": "Replaced faulty HDMI cable. Signal strength increased from 65% to 92%. Scheduled technician visit waived.", "category": "technical", "agent_name": "tech_fahad"},
#   {"query": "Internet drops at 8PM daily", "resolution": "Identified neighborhood congestion. Recommended: 1) Use QoS settings 2) Schedule updates after midnight 3) Offered 10% discount for 3 months.", "category": "internet", "agent_name": "tech_abdul"},
#   {"query": "Double charged for international SMS", "resolution": "Confirmed system error for UAE SMS. Issued 75 SAR credit and updated rate table. Advised to monitor next bill.", "category": "billing", "agent_name": "billing_lina"},
#   {"query": "5G not working in new iPhone", "resolution": "APN settings needed update to 'stc5g'. Guided through: Settings > Cellular > Network Selection > Manual STC 5G.", "category": "technical", "agent_name": "support_khalid"},
#   {"query": "Business account transfer delay", "resolution": "Missing notarized authorization. Faxed template to customer. Process completed 2 hours after document receipt.", "category": "account", "agent_name": "biz_mohammed"},
#   {"query": "No service in basement office", "resolution": "Recommended WiFi calling or femtocell (provided at 50% discount). Signal booster incompatible with 5G frequencies.", "category": "general", "agent_name": "tech_salim"},
#   {"query": "YouTube buffering on fiber", "resolution": "Changed DNS to 8.8.8.8/8.8.4.4. Disabled IPv6. Verified YouTube server peering was congested - escalated to network team.", "category": "internet", "agent_name": "tech_noura"},
#   {"query": "VAT charged on exempt entity", "resolution": "Verified tax exemption expired. Assisted with renewal and processed 1,200 SAR refund. Added calendar reminder for next year.", "category": "billing", "agent_name": "billing_huda"},
#   {"query": "STC TV remote unresponsive", "resolution": "Paired new remote (delivered next day). Identified IR sensor obstruction in 30% of similar cases.", "category": "technical", "agent_name": "support_fahad"},
#   {"query": "Hijacked WhatsApp number", "resolution": "Immediate SIM block. Guided through WhatsApp recovery via email. Recommended enabling 2FA on all linked accounts.", "category": "account", "agent_name": "security_amir"},
#   {"query": "Hajj package activation failed", "resolution": "Location services needed enabling. Manually activated package after GPS confirmation in Makkah.", "category": "general", "agent_name": "support_layla"},
#   {"query": "Gaming latency spikes", "resolution": "Recommended: 1) Wired connection 2) Port forwarding 3) Gamer Pro package. Reduced ping from 180ms to 45ms.", "category": "internet", "agent_name": "tech_gamer"},
#   {"query": "Early termination fee dispute", "resolution": "Waived 50% fee after verifying customer relocated to non-service area. Required proof of new address.", "category": "billing", "agent_name": "retention_yousef"},
#   {"query": "Router overheating", "resolution": "Replaced under warranty. Recommended: 1) Ventilated area 2) Scheduled reboots 3) Firmware update to v2.1.5.", "category": "technical", "agent_name": "tech_karim"},
#   {"query": "Corporate bulk SMS rejection", "resolution": "Missing content registry approval. Assisted with submission. Temporary whitelist applied during 48h approval process.", "category": "account", "agent_name": "biz_sara"},
#   {"query": "No incoming calls abroad", "resolution": "Roaming bar accidentally enabled. Disabled restriction and credited 20 SAR for inconvenience.", "category": "general", "agent_name": "support_nada"},
#   {"query": "Fiber installation damage", "resolution": "Coordinated with contractor for wall repair. Compensated 300 SAR and provided 3 months free service.", "category": "internet", "agent_name": "ops_hamad"},
#   {"query": "Employee personal data usage", "resolution": "Identified compromised credentials. Reset all admin passwords and implemented IP restriction for HR department.", "category": "account", "agent_name": "security_khalid"},
#   {"query": "Duplicate payment refund", "resolution": "Processed same-day refund via original payment method. System flagged duplicate transaction ID.", "category": "billing", "agent_name": "billing_ali"},
#   {"query": "4K channels pixelating", "resolution": "Upgraded HDMI cable and adjusted TV settings to 2160p 60Hz. Signal strength was adequate (87%).", "category": "technical", "agent_name": "tv_technician"},
#   {"query": "Prepaid balance disappearance", "resolution": "Identified system error during migration. Restored balance + 20% bonus. New balance expiration extended to 90 days.", "category": "general", "agent_name": "prepaid_salma"}
# ]

# headers = {
#   'Content-Type': 'application/json'
# }

# for i in payload:
#     response = requests.request("POST", url, headers=headers, data=json.dumps(i))
#     print(response.text)

# tickets


url = "http://localhost:8000/tickets"

payload = [
  {"title": "Internet outage in Al-Olaya", "description": "No internet connection since morning. Router lights are red. Working from home is impossible!", "customer_email": "ahmed@example.com", "customer_name": "Ahmed Al-Saud", "location": {  "latitude": 24.7136,   "longitude": 46.6753}},
  {
    "title": "Poor signal in Jeddah Corniche",
    "description": "Can't make calls near the waterfront. Signal drops constantly during business meetings.",
    "customer_email": "leila@domain.sa",
    "customer_name": "Leila Hassan",
    "location": {
      "latitude": 21.5925,
      "longitude": 39.1767
    }
  },
  {
    "title": "Billing discrepancy",
    "description": "Charged for international calls I never made. Need detailed call logs.",
    "customer_email": "khalid@mail.sa",
    "customer_name": "Khalid Abadi",
    "location": {
      "latitude": 24.4686,
      "longitude": 39.6142
    }
  },
  {
    "title": "TV channel package missing",
    "description": "Sports channels disappeared after system update. Missing crucial matches!",
    "customer_email": "faisal@provider.sa",
    "customer_name": "Faisal Rashed",
    "location": {
      "latitude": 26.4202,
      "longitude": 50.0888
    }
  },
  {
    "title": "Slow 5G in Riyadh Center",
    "description": "5G speed slower than my old 4G. Speedtest shows 12Mbps near Kingdom Tower.",
    "customer_email": "nora@company.sa",
    "customer_name": "Nora Al-Faisal",
    "location": {
      "latitude": 24.7113,
      "longitude": 46.6750
    }
  },
  {
    "title": "Voicemail not working",
    "description": "Callers can't leave messages. Get 'mailbox full' error when it's empty.",
    "customer_email": "yousef@contact.sa",
    "customer_name": "Yousef Omar",
    "location": {
      "latitude": 21.5433,
      "longitude": 39.1728
    }
  },
  {
    "title": "Fiber installation delay",
    "description": "Appointment missed 3 times. Technician never arrives. Urgent for home office.",
    "customer_email": "sara@user.sa",
    "customer_name": "Sara Khaled",
    "location": {
      "latitude": 24.8224,
      "longitude": 46.6390
    }
  },
  {
    "title": "International roaming failure",
    "description": "No service in Dubai despite premium package. Incurred massive charges.",
    "customer_email": "tariq@client.sa",
    "customer_name": "Tariq Mansour",
    "location": {
      "latitude": 24.7262,
      "longitude": 46.6644
    }
  },
  {
    "title": "WiFi disconnects repeatedly",
    "description": "Connection drops every 15 minutes. Router restart only fixes temporarily.",
    "customer_email": "lama@customer.sa",
    "customer_name": "Lama Abdullah",
    "location": {
      "latitude": 21.4239,
      "longitude": 39.8255
    }
  },
  {
    "title": "SMS delivery failure",
    "description": "Bank OTPs not received. Confirmed bank is sending but nothing arrives.",
    "customer_email": "majid@inbox.sa",
    "customer_name": "Majid Saleh",
    "location": {
      "latitude": 24.4667,
      "longitude": 39.6000
    }
  },
  {
    "title": "💡 My SIM card thinks it's a falcon",
    "description": "Every time I try to call, it flies away from my phone. Possibly identifies as migratory bird?",
    "customer_email": "badr@joke.sa",
    "customer_name": "Badr Al-Johani",
    "location": {
      "latitude": 24.6951,
      "longitude": 46.7298
    }
  },
  {
    "title": "💡 Phone only works during iftar",
    "description": "Perfect signal at sunset but dead otherwise. Ramadan miracle or tech glitch?",
    "customer_email": "omar@funny.sa",
    "customer_name": "Omar Zahrani",
    "location": {
      "latitude": 21.3891,
      "longitude": 39.8579
    }
  },
  {
    "title": "💡 My router brews coffee",
    "description": "Started dispensing arabica instead of WiFi. Good coffee but terrible bandwidth.",
    "customer_email": "nada@silly.sa",
    "customer_name": "Nada Al-Rajhi",
    "location": {
      "latitude": 24.6988,
      "longitude": 46.7215
    }
  },
  {
    "title": "💡 5G turned cat into hologram",
    "description": "Fluffy now projects 3D images around house. Cute but scaring the kids.",
    "customer_email": "hassan@lol.sa",
    "customer_name": "Hassan Qasim",
    "location": {
      "latitude": 24.6255,
      "longitude": 46.7076
    }
  },
  {
    "title": "💡 Phone autocorrects to camel jokes",
    "description": "'Meeting rescheduled' becomes 'camel rescheduled'. Important texts sound like desert humor.",
    "customer_email": "fatima@haha.sa",
    "customer_name": "Fatima Al-Harbi",
    "location": {
      "latitude": 24.8011,
      "longitude": 46.6383
    }
  },
  {
    "title": "💡 Bill paid in sand dollars",
    "description": "Tried to settle invoice with beach sand. Now router only streams desert documentaries.",
    "customer_email": "khalid@witty.sa",
    "customer_name": "Khalid Dossary",
    "location": {
      "latitude": 26.3021,
      "longitude": 50.1520
    }
  },
  {
    "title": "💡 SIM card sings nasheeds",
    "description": "Instead of dial tone, hear religious songs. Beautiful but can't make calls.",
    "customer_email": "salma@music.sa",
    "customer_name": "Salma Al-Ghamdi",
    "location": {
      "latitude": 21.6245,
      "longitude": 39.2067
    }
  },
  {
    "title": "💡 Phone thinks it's a date palm",
    "description": "Keeps asking for fertilizer notifications. Requesting tech support with agricultural degree.",
    "customer_email": "rayan@palmtree.sa",
    "customer_name": "Rayan Al-Mutairi",
    "location": {
      "latitude": 24.4738,
      "longitude": 39.6161
    }
  },
  {
    "title": "💡 Mobile data only works for memes",
    "description": "Can stream funny cat videos 4K, but emails won't load. Priorities messed up.",
    "customer_email": "zain@meme.sa",
    "customer_name": "Zain Bashir",
    "location": {
      "latitude": 24.7600,
      "longitude": 46.6600
    }
  },
  {
    "title": "💡 Phone auto-dials the Crown Prince",
    "description": "Accidentally called MBS 17 times while ordering kabsa. Very embarrassing.",
    "customer_email": "fahad@oops.sa",
    "customer_name": "Fahad Al-Shehri",
    "location": {
      "latitude": 24.7111,
      "longitude": 46.6742
    }
  },
  {
    "title": "💡 Router emits oud fragrance",
    "description": "Smells like a music festival but no internet. Scented troubleshooting needed.",
    "customer_email": "layla@smell.sa",
    "customer_name": "Layla Al-Qahtani",
    "location": {
      "latitude": 21.4932,
      "longitude": 39.1937
    }
  },
  {
    "title": "Can't activate eSIM",
    "description": "QR code scan fails repeatedly. Need manual activation support.",
    "customer_email": "amir@business.sa",
    "customer_name": "Amir Hassan",
    "location": {
      "latitude": 24.7883,
      "longitude": 46.6496
    }
  },
  {
    "title": "Landline static noise",
    "description": "Constant buzzing on home phone. Can't hear callers properly.",
    "customer_email": "haya@residence.sa",
    "customer_name": "Haya Mohammed",
    "location": {
      "latitude": 24.7288,
      "longitude": 46.6621
    }
  },
  {
    "title": "TV remote unresponsive",
    "description": "New IR remote won't pair with set-top box. Batteries replaced.",
    "customer_email": "saad@techhelp.sa",
    "customer_name": "Saad Al-Zahrani",
    "location": {
      "latitude": 26.3516,
      "longitude": 50.1974
    }
  },
  {
    "title": "Duplicate SMS received",
    "description": "Get every text message 5-7 times. Flooding my inbox.",
    "customer_email": "noura@mobile.sa",
    "customer_name": "Noura Fahad",
    "location": {
      "latitude": 24.7012,
      "longitude": 46.6783
    }
  },
  {
    "title": "Email not syncing on mobile",
    "description": "@company.sa enterprise emails not updating on phone. Works on desktop.",
    "customer_email": "waleed@corp.sa",
    "customer_name": "Waleed Akram",
    "location": {
      "latitude": 24.7225,
      "longitude": 46.6873
    }
  },
  {
    "title": "Frequent call drops in Dammam",
    "description": "Calls disconnect after 2 minutes near Corniche. Consistent issue.",
    "customer_email": "ibrahim@dhahran.sa",
    "customer_name": "Ibrahim Saleem",
    "location": {
      "latitude": 26.4362,
      "longitude": 50.1033
    }
  },
  {
    "title": "Wrong name on bill",
    "description": "Account shows 'Invisible Man' instead of my name. Identity crisis!",
    "customer_email": "kareem@billing.sa",
    "customer_name": "Kareem Jassim",
    "location": {
      "latitude": 24.4922,
      "longitude": 39.7062
    }
  },
  {
    "title": "Mobile hotspot limit error",
    "description": "Shows 'data exhausted' when 80% remains. Can't tether laptop.",
    "customer_email": "dalal@student.sa",
    "customer_name": "Dalal Ahmed",
    "location": {
      "latitude": 24.8137,
      "longitude": 46.6524
    }
  },
  {
    "title": "VoIP quality issues",
    "description": "Conference calls echo terribly. Participants sound underwater.",
    "customer_email": "yara@office.sa",
    "customer_name": "Yara Tawfiq",
    "location": {
      "latitude": 24.6995,
      "longitude": 46.6851
    }
  },
  {
    "title": "Security alert false positives",
    "description": "Get 'suspicious activity' texts every hour. No unusual logins.",
    "customer_email": "faisal@secure.sa",
    "customer_name": "Faisal Nasser",
    "location": {
      "latitude": 24.7701,
      "longitude": 46.6627
    }
  },
  {
    "title": "Can't unsubscribe from promotions",
    "description": "Opted out 9 times but still get marketing SMS. Violating preferences.",
    "customer_email": "maha@stop.sa",
    "customer_name": "Maha Khalid",
    "location": {
      "latitude": 21.4211,
      "longitude": 39.8262
    }
  },
  {
    "title": "Fiber cable exposed",
    "description": "Construction damaged line near my villa. Open wire hazard.",
    "customer_email": "rashid@villa.sa",
    "customer_name": "Rashid Al-Mansour",
    "location": {
      "latitude": 24.8019,
      "longitude": 46.6417
    }
  },
  {
    "title": "International SMS blocked",
    "description": "Can't text UAE numbers. Error: 'destination restricted'.",
    "customer_email": "saeed@family.sa",
    "customer_name": "Saeed Omar",
    "location": {
      "latitude": 26.2934,
      "longitude": 50.1980
    }
  },
  {
    "title": "Set-top box freezing",
    "description": "TV freezes during prime time shows. Requires daily reboot.",
    "customer_email": "lina@entertain.sa",
    "customer_name": "Lina Abdul",
    "location": {
      "latitude": 24.7128,
      "longitude": 46.6719
    }
  },
  {
    "title": "App login failure",
    "description": "'MyAccount' app rejects valid credentials. Web login works fine.",
    "customer_email": "turkia@appuser.sa",
    "customer_name": "Turkia Ali",
    "location": {
      "latitude": 21.5985,
      "longitude": 39.2103
    }
  },
  {
    "title": "Data usage spike alert",
    "description": "Received 95% usage warning but barely used phone. Suspect error.",
    "customer_email": "mohsen@data.sa",
    "customer_name": "Mohsen Kamal",
    "location": {
      "latitude": 24.4715,
      "longitude": 39.6119
    }
  },
  {
    "title": "Port request delayed",
    "description": "Number porting stuck 'in process' for 12 days. Can't receive calls.",
    "customer_email": "naser@switch.sa",
    "customer_name": "Naser Idris",
    "location": {
      "latitude": 24.8200,
      "longitude": 46.6400
    }
  },
  {
    "title": "WiFi dead zones in home",
    "description": "No signal in kitchen and bedrooms. House layout issue?",
    "customer_email": "salah@home.sa",
    "customer_name": "Salah Rajab",
    "location": {
      "latitude": 24.7023,
      "longitude": 46.6921
    }
  },
  {
    "title": "Unrecognized device on network",
    "description": "Unknown device named 'Ghost' connected to home WiFi. Security concern.",
    "customer_email": "farah@securehome.sa",
    "customer_name": "Farah Musa",
    "location": {
      "latitude": 24.7900,
      "longitude": 46.6500
    }
  },
  {
    "title": "Premium channels pixelated",
    "description": "Paid movie channels look like 8-bit video games. Quality unacceptable.",
    "customer_email": "basim@tvsub.sa",
    "customer_name": "Basim Karim",
    "location": {
      "latitude": 26.3315,
      "longitude": 50.1820
    }
  },
  {
    "title": "Auto-payment failure",
    "description": "Credit card charge declined despite valid details. Service suspension threat.",
    "customer_email": "jamal@autopay.sa",
    "customer_name": "Jamal Fahd",
    "location": {
      "latitude": 24.7456,
      "longitude": 46.6632
    }
  },
  {
    "title": "Replacement SIM not activated",
    "description": "Lost phone got new SIM, still shows 'no service' after 48 hours.",
    "customer_email": "arwa@newsim.sa",
    "customer_name": "Arwa Salem",
    "location": {
      "latitude": 21.4858,
      "longitude": 39.1925
    }
  },
  {
    "title": "Parental control bypassed",
    "description": "Kids accessed blocked gaming sites. Filters not working.",
    "customer_email": "hamad@dad.sa",
    "customer_name": "Hamad Zayed",
    "location": {
      "latitude": 24.7150,
      "longitude": 46.6720
    }
  },
  {
    "title": "No 4G in Abha mountains",
    "description": "Only edge network in tourist area. Coverage map showed full bars.",
    "customer_email": "khalil@travel.sa",
    "customer_name": "Khalil Asiri",
    "location": {
      "latitude": 18.2305,
      "longitude": 42.5001
    }
  },
  {
    "title": "Modem overheating",
    "description": "Router too hot to touch. Smells like burning plastic. Fire hazard?",
    "customer_email": "sirine@device.sa",
    "customer_name": "Sirine Malik",
    "location": {
      "latitude": 24.7089,
      "longitude": 46.6687
    }
  },
  {
    "title": "Call forwarding loops",
    "description": "Forwarded calls bounce between numbers endlessly. Can't answer any.",
    "customer_email": "muneer@forward.sa",
    "customer_name": "Muneer Adnan",
    "location": {
      "latitude": 24.6982,
      "longitude": 46.7203
    }
  },
  {
    "title": "Late payment fee dispute",
    "description": "Charged late fee despite paying on due date. Need waiver.",
    "customer_email": "hussain@payissue.sa",
    "customer_name": "Hussain Qureshi",
    "location": {
      "latitude": 24.4700,
      "longitude": 39.6100
    }
  },
  {
    "title": "Business landline down",
    "description": "Office phone dead for 2 days. Affecting customer support.",
    "customer_email": "zeina@business.sa",
    "customer_name": "Zeina Akbar",
    "location": {
      "latitude": 24.7200,
      "longitude": 46.6900
    }
  },
  {
    "title": "Fiber optic cable damage in Al-Nakheel",
    "description": "Construction work severed underground fiber line. Entire neighborhood offline.",
    "customer_email": "khalid_almansour@business.sa",
    "customer_name": "Khalid Al-Mansour",
    "location": {
      "latitude": 24.7804,
      "longitude": 46.6972
    }
  },
  {
    "title": "Persistent packet loss in King Abdullah Financial District",
    "description": "20-30% packet loss during business hours affecting VoIP systems.",
    "customer_email": "sara_alghamdi@finance.sa",
    "customer_name": "Sara Al-Ghamdi",
    "location": {
      "latitude": 24.7602,
      "longitude": 46.6428
    }
  },
  {
    "title": "Billing system double-charging for international roaming",
    "description": "Duplicate charges for UAE usage appear on November invoice. Transaction IDs: INV-78945, INV-78946",
    "customer_email": "omar_nasser@corporate.sa",
    "customer_name": "Omar Nasser",
    "location": {
      "latitude": 24.7136,
      "longitude": 46.6753
    }
  },
  {
    "title": "5G tower malfunction in Dhahran",
    "description": "Sector 3 offline causing coverage gaps in residential zone 7.",
    "customer_email": "fahad_khouri@dhahran.sa",
    "customer_name": "Fahad Khouri",
    "location": {
      "latitude": 26.2915,
      "longitude": 50.1583
    }
  },
  {
    "title": "Enterprise VPN connectivity failure",
    "description": "Unable to establish secure connection to corporate network since system update.",
    "customer_email": "itadmin@alrajhi-group.sa",
    "customer_name": "Majid Al-Rajhi",
    "location": {
      "latitude": 24.6987,
      "longitude": 46.7219
    }
  },
  {
    "title": "DSLAM equipment overheating in Jeddah Central",
    "description": "Temperature alerts at exchange building JB-07. Requires urgent maintenance.",
    "customer_email": "tech_officer@jeddahmunicipal.sa",
    "customer_name": "Yousef Hassan",
    "location": {
      "latitude": 21.5433,
      "longitude": 39.1728
    }
  },
  {
    "title": "SMS gateway failure for banking alerts",
    "description": "Financial institutions report undelivered transaction notifications.",
    "customer_email": "nabil.ahmed@ncb.sa",
    "customer_name": "Nabil Ahmed",
    "location": {
      "latitude": 24.7105,
      "longitude": 46.6732
    }
  },
  {
    "title": "Fiber backbone outage between Riyadh and Qassim",
    "description": "Backhaul link down causing regional service degradation. Ticket: INC-789456",
    "customer_email": "noc_engineer@telecom.sa",
    "customer_name": "Abdullah Faisal",
    "location": {
      "latitude": 25.3548,
      "longitude": 43.5543
    }
  },
  {
    "title": "Incorrect service activation in Al-Khobar",
    "description": "Order #789123 for 500Mbps fiber installed as 100Mbps package.",
    "customer_email": "salem@alghurairgroup.sa",
    "customer_name": "Salem Al-Ghurair",
    "location": {
      "latitude": 26.2796,
      "longitude": 50.2082
    }
  },
  {
    "title": "Mobile number porting failure",
    "description": "Number 05XXXXXXXX stuck in transfer status for 72 hours.",
    "customer_email": "layla.omar@customer.sa",
    "customer_name": "Layla Omar",
    "location": {
      "latitude": 24.8224,
      "longitude": 46.6390
    }
  },
  {
    "title": "VoLTE compatibility issues with Samsung S23 series",
    "description": "Calls drop when switching between 5G and VoLTE. Multiple customers reporting.",
    "customer_email": "mobile_support@telecom.sa",
    "customer_name": "Technical Support Team",
    "location": {
      "latitude": 24.7136,
      "longitude": 46.6753
    }
  },
  {
    "title": "DNS resolution failure in Eastern Province",
    "description": "Recursive DNS servers not responding to queries. Affecting domain.sa sites.",
    "customer_email": "admin@dammamhospital.sa",
    "customer_name": "Dr. Ahmed Mansouri",
    "location": {
      "latitude": 26.3921,
      "longitude": 50.0759
    }
  },
  {
    "title": "Faulty ONT in Al-Ahsa District",
    "description": "Optical Network Terminal requires replacement. Flashing failure light.",
    "customer_email": "mohammad@alhassavilla.sa",
    "customer_name": "Mohammad Al-Hassan",
    "location": {
      "latitude": 25.3892,
      "longitude": 49.5869
    }
  },
  {
    "title": "Underground conduit flooding in Jeddah",
    "description": "Water infiltration in cable ducts after heavy rainfall. Potential short circuit risk.",
    "customer_email": "facilities@jeddahport.sa",
    "customer_name": "Khalid Al-Zahrani",
    "location": {
      "latitude": 21.5433,
      "longitude": 39.1728
    }
  },
  {
    "title": "Corporate account management portal down",
    "description": "Enterprise login portal returning 503 errors for 12+ hours.",
    "customer_email": "it_director@aramco.sa",
    "customer_name": "Faisal Al-Sheikh",
    "location": {
      "latitude": 24.7111,
      "longitude": 46.6742
    }
  },
  {
    "title": "Microwave link interference in Tabuk",
    "description": "New construction causing signal degradation on tower TBK-12.",
    "customer_email": "network_ops@telecom.sa",
    "customer_name": "Regional Operations",
    "location": {
      "latitude": 28.3835,
      "longitude": 36.5662
    }
  },
  {
    "title": "Incorrect tax calculation on business accounts",
    "description": "VAT applied at 15% instead of 5% for enterprise customers.",
    "customer_email": "finance@alswalimgroup.sa",
    "customer_name": "Noura Al-Swalim",
    "location": {
      "latitude": 24.7225,
      "longitude": 46.6873
    }
  },
  {
    "title": "Fiber splice damage in Medina Central",
    "description": "Vandalism incident affecting distribution node MD-45.",
    "customer_email": "security@medinamunicipal.sa",
    "customer_name": "Captain Rashid",
    "location": {
      "latitude": 24.4686,
      "longitude": 39.6142
    }
  },
  {
    "title": "IoT SIM provisioning failure",
    "description": "Bulk activation of 500+ industrial IoT SIMs stalled at pending status.",
    "customer_email": "iot_manager@sec.sa",
    "customer_name": "Amira Taha",
    "location": {
      "latitude": 24.7600,
      "longitude": 46.6600
    }
  },
  {
    "title": "Peak-hour congestion in Riyadh Diplomatic Quarter",
    "description": "Evening bandwidth drops below 10Mbps for premium business accounts.",
    "customer_email": "embassy_tech@franceksa.sa",
    "customer_name": "Pierre Dubois",
    "location": {
      "latitude": 24.6881,
      "longitude": 46.6254
    }
  },
  {
    "title": "Faulty meter readings in Al-Kharj",
    "description": "Usage data not transmitting from remote meters. Affecting billing cycle.",
    "customer_email": "utilities@alkharjcity.sa",
    "customer_name": "Municipal Services Dept",
    "location": {
      "latitude": 24.1554,
      "longitude": 47.3346
    }
  },
  {
    "title": "Emergency services priority access failure",
    "description": "Hospital communications not receiving network priority during congestion.",
    "customer_email": "comms@kfsh.sa",
    "customer_name": "Dr. Leila Abadi",
    "location": {
      "latitude": 24.6988,
      "longitude": 46.7215
    }
  },
  {
    "title": "Fiber termination panel malfunction",
    "description": "Central office rack F3-P8 offline in Dammam exchange.",
    "customer_email": "co_engineer@telecom.sa",
    "customer_name": "Hassan Qasim",
    "location": {
      "latitude": 26.4202,
      "longitude": 50.0888
    }
  },
  {
    "title": "Directory assistance database corruption",
    "description": "188 service returning incorrect business listings.",
    "customer_email": "directory_support@telecom.sa",
    "customer_name": "Service Operations",
    "location": {
      "latitude": 24.7136,
      "longitude": 46.6753
    }
  },
  {
    "title": "LTE-A carrier aggregation failure",
    "description": "Unable to bond channels in high-density areas. Speed capped at 100Mbps.",
    "customer_email": "rf_engineer@telecom.sa",
    "customer_name": "Ahmed Farhan",
    "location": {
      "latitude": 24.7113,
      "longitude": 46.6750
    }
  },
  {
    "title": "Missed service level agreement - business fiber",
    "description": "48-hour repair guarantee breached for enterprise account #789456.",
    "customer_email": "contracts@alhabibgroup.sa",
    "customer_name": "Kareem Al-Habib",
    "location": {
      "latitude": 24.7256,
      "longitude": 46.6647
    }
  },
  {
    "title": "Generator fuel supply interruption - remote tower",
    "description": "Tower ABH-08 running on backup batteries due to fuel delivery delay.",
    "customer_email": "facilities@towerco.sa",
    "customer_name": "Tower Maintenance",
    "location": {
      "latitude": 18.2305,
      "longitude": 42.5001
    }
  },
  {
    "title": "Fiber mislabeling during new development installation",
    "description": "Cross-connected lines in Al-Faisaliah Gardens causing wrong premise activations.",
    "customer_email": "project_mgr@emaar.sa",
    "customer_name": "Abdul Rahman",
    "location": {
      "latitude": 24.7324,
      "longitude": 46.6389
    }
  },
  {
    "title": "Call center IVR system outage",
    "description": "Automated customer service menu non-responsive. Customers unable to reach support.",
    "customer_email": "callcenter_mgr@telecom.sa",
    "customer_name": "Maha Al-Sudairi",
    "location": {
      "latitude": 24.7136,
      "longitude": 46.6753
    }
  },
  {
    "title": "Microwave alignment drift - Red Sea coastal stations",
    "description": "Atmospheric conditions causing link instability between Jeddah and Yanbu.",
    "customer_email": "transmission@telecom.sa",
    "customer_name": "Transmission Dept",
    "location": {
      "latitude": 22.3125,
      "longitude": 39.1028
    }
  },
  {
    "title": "Unauthorized SIM swap fraud",
    "description": "Customer reports number ported without authorization. Account #7890123",
    "customer_email": "fraud_department@telecom.sa",
    "customer_name": "Security Division",
    "location": {
      "latitude": 24.7136,
      "longitude": 46.6753
    }
  },
  {
    "title": "BTS cabinet break-in attempt in Najran",
    "description": "Security breach at site NJR-14. Equipment tampering suspected.",
    "customer_email": "site_security@telecom.sa",
    "customer_name": "Field Operations",
    "location": {
      "latitude": 17.5656,
      "longitude": 44.2289
    }
  },
  {
    "title": "Optical signal degradation on GCC backbone",
    "description": "BER exceeding thresholds on segment Dammam-Riyadh. OTDR trace required.",
    "customer_email": "backbone_ops@telecom.sa",
    "customer_name": "Network Integrity",
    "location": {
      "latitude": 25.3548,
      "longitude": 43.5543
    }
  },
  {
    "title": "In-building coverage system failure - Riyadh Metro",
    "description": "Distributed antenna system offline at Qasr Al-Hokm station.",
    "customer_email": "metro_comms@riyadh.sa",
    "customer_name": "Metro Operations",
    "location": {
      "latitude": 24.6284,
      "longitude": 46.7157
    }
  },
  {
    "title": "Payments processing delay",
    "description": "Bank transfers not reflecting in accounts for 72+ hours. Transaction batch #789123",
    "customer_email": "billing_system@telecom.sa",
    "customer_name": "Billing Department",
    "location": {
      "latitude": 24.7136,
      "longitude": 46.6753
    }
  },
  {
    "title": "GPON OLT card failure in Makkah",
    "description": "Slot 3 Card 2 offline affecting 500+ subscribers.",
    "customer_email": "gpon_tech@telecom.sa",
    "customer_name": "Fiber Network Team",
    "location": {
      "latitude": 21.3891,
      "longitude": 39.8579
    }
  },
  {
    "title": "Temporary number allocation error",
    "description": "New activations assigning duplicate mobile numbers.",
    "customer_email": "number_pool@telecom.sa",
    "customer_name": "Number Administration",
    "location": {
      "latitude": 24.7136,
      "longitude": 46.6753
    }
  },
  {
    "title": "Weather-related tower damage in Asir Province",
    "description": "High winds toppled antenna arrays at site ASR-22.",
    "customer_email": "disaster_recovery@telecom.sa",
    "customer_name": "Emergency Response",
    "location": {
      "latitude": 18.2305,
      "longitude": 42.5001
    }
  },
  {
    "title": "Voicemail system corruption",
    "description": "Customers report deleted messages and greeting resets.",
    "customer_email": "voice_services@telecom.sa",
    "customer_name": "Voice Platform Team",
    "location": {
      "latitude": 24.7136,
      "longitude": 46.6753
    }
  },
  {
    "title": "Spectrum interference in 2100MHz band - Eastern Region",
    "description": "Unidentified signal causing dropped calls. Requires spectrum analysis.",
    "customer_email": "rf_engineering@telecom.sa",
    "customer_name": "Radio Frequency Dept",
    "location": {
      "latitude": 26.3021,
      "longitude": 50.1520
    }
  },
  {
    "title": "Fiber splicing trailer accident on Hail Highway",
    "description": "Service vehicle collision damaged critical splicing equipment.",
    "customer_email": "field_ops@telecom.sa",
    "customer_name": "Northern Region Ops",
    "location": {
      "latitude": 27.5114,
      "longitude": 41.7208
    }
  },
  {
    "title": "Data center cooling system failure",
    "description": "CRAC unit shutdown in Riyadh DC-3 causing temperature rise.",
    "customer_email": "datacenter@telecom.sa",
    "customer_name": "Facilities Management",
    "location": {
      "latitude": 24.7250,
      "longitude": 46.6380
    }
  },
  {
    "title": "Billing system migration data corruption",
    "description": "Legacy account data not transferring correctly to new platform.",
    "customer_email": "billing_migration@telecom.sa",
    "customer_name": "Billing Project Team",
    "location": {
      "latitude": 24.7136,
      "longitude": 46.6753
    }
  },
  {
    "title": "Submarine cable maintenance notification failure",
    "description": "Customers not receiving scheduled outage alerts for FALCON cable work.",
    "customer_email": "customer_comms@telecom.sa",
    "customer_name": "Customer Communications",
    "location": {
      "latitude": 26.3516,
      "longitude": 50.1974
    }
  },
  {
    "title": "Industrial IoT latency spikes",
    "description": "M2M communications experiencing 500ms+ latency in Yanbu Industrial City.",
    "customer_email": "iot_operations@sabic.sa",
    "customer_name": "SABIC Automation",
    "location": {
      "latitude": 23.9925,
      "longitude": 38.2310
    }
  },
  {
    "title": "Authorization failure for business account changes",
    "description": "Approved service modifications not implementing in provisioning system.",
    "customer_email": "enterprise_support@telecom.sa",
    "customer_name": "Business Services",
    "location": {
      "latitude": 24.7136,
      "longitude": 46.6753
    }
  },
  {
    "title": "Copper pair degradation in historic Jeddah",
    "description": "ADSL services failing due to corroded legacy cabling.",
    "customer_email": "heritage@jeddah.sa",
    "customer_name": "Historic District Office",
    "location": {
      "latitude": 21.4858,
      "longitude": 39.1925
    }
  },
  {
    "title": "Network time protocol server desynchronization",
    "description": "Device clocks drifting across network elements. Stratum 1 server unreachable.",
    "customer_email": "network_timing@telecom.sa",
    "customer_name": "Precision Timing",
    "location": {
      "latitude": 24.7136,
      "longitude": 46.6753
    }
  },
  {
    "title": "Supply chain delay for Huawei equipment",
    "description": "Critical 5G expansion components held at customs. Project delay imminent.",
    "customer_email": "network_expansion@telecom.sa",
    "customer_name": "Network Deployment",
    "location": {
      "latitude": 24.7136,
      "longitude": 46.6753
    }
  },
  {
    "title": "Fiber optic sensor false alarms",
    "description": "Security monitoring system triggering false positives along pipeline route.",
    "customer_email": "pipeline_security@aramco.sa",
    "customer_name": "Aramco Security",
    "location": {
      "latitude": 26.4362,
      "longitude": 50.1033
    }
  },
  {
    "title": "Call detail record processing backlog",
    "description": "CDR files not processing for billing. 12-hour delay accumulating.",
    "customer_email": "mediation@billing.sa",
    "customer_name": "Billing Systems",
    "location": {
      "latitude": 24.7136,
      "longitude": 46.6753
    }
  },
  {
    "title": "Emergency alert system test failure",
    "description": "National warning system not broadcasting to target areas during scheduled test.",
    "email": "public_safety@mcit.sa",
    "customer_name": "MCIT Emergency Comms",
    "location": {
      "latitude": 24.7136,
      "longitude": 46.6753
    }
  }
]

headers = {
  'Content-Type': 'application/json'
}

for i in payload:
    response = requests.request("POST", url, headers=headers, data=json.dumps(i))
    print(response.text)