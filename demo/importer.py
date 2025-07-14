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
  {"title": "Unexpected roaming charges", "description": "Charged 350 SAR for UAE roaming despite activating package. Was only there for 3 days!", "customer_email": "sara@example.com", "customer_name": "Sara Mohammed", "location": {  "latitude": 21.5433,   "longitude": 39.1728}},
  {"title": "5G not working in Dammam", "description": "Phone shows 5G but speed is like 3G. Tried different locations in city center. iPhone 14 Pro.", "customer_email": "khalid@example.com", "customer_name": "Khalid Hassan", "location": {  "latitude": 26.4207,   "longitude": 50.0888}},
  {"title": "Can't pay bill online", "description": "Payment fails on STC Pay app with 'processor error'. Tried 3 cards that work elsewhere.", "customer_email": "fatima@example.com", "customer_name": "Fatima Abdullah"},
  {"title": "TV package missing channels", "description": "Sports package channels 120-125 not showing after renewal. Restarted box twice.", "customer_email": "omar@example.com", "customer_name": "Omar Ibrahim"},
  {"title": "Business account transfer", "description": "Need to transfer 5 numbers from father's account to my business. Documents ready.", "customer_email": "nora@example.com", "customer_name": "Nora Al-Faisal", "location": {  "latitude": 24.7743,   "longitude": 46.7382}},
  {"title": "Fiber installation delay", "description": "Appointment missed 3 times! Promised 48h installation still not done after 2 weeks.", "customer_email": "faisal@example.com", "customer_name": "Faisal Rashid", "location": {  "latitude": 24.5247,   "longitude": 39.5692}},
  {"title": "International calls blocked", "description": "Suddenly can't call Egypt numbers. Account shows 'restricted'. No notification.", "customer_email": "layla@example.com", "customer_name": "Layla Mahmoud"},
  {"title": "Data breach concern", "description": "Received SMS with OTP I didn't request. Worried about account security.", "customer_email": "majid@example.com", "customer_name": "Majid Sultan"},
  {"title": "Corporate plan upgrade", "description": "Need to upgrade 50-employee plan to unlimited 5G. Requesting manager approval.", "customer_email": "salim@example.com", "customer_name": "Salim Corporation"},

  {"title": "Fiber outage in Al-Malaz", "description": "No internet since yesterday evening. ONT shows red LOS light. Neighbors also affected.", "customer_email": "majid.alsaud@example.com", "customer_name": "Majid Al-Saud", "location": {"latitude": 24.6408, "longitude": 46.7728}},
  {"title": "International roaming not working in Turkey", "description": "Paid for Europe package but only getting 'Emergency Calls Only'. Need urgent fix for business trip!", "customer_email": "business.traveler@example.com", "customer_name": "Abdullah Trading Co", "location": {"latitude": 24.7136, "longitude": 46.6753}},
  {"title": "STC TV app login failure", "description": "Error 'Account not recognized' on new Smart TV. Works on phone but not Samsung TV.", "customer_email": "entertainment@example.com", "customer_name": "Family Al-Rashid"},
  {"title": "Unauthorized SIM swap", "description": "Received SMS about new SIM activation. Didn't request this! Can't access my number now.", "customer_email": "security.concern@example.com", "customer_name": "Lina Ahmed", "location": {"latitude": 21.5433, "longitude": 39.1728}},
  {"title": "Business plan upgrade request", "description": "Need to move 35 employees from Consumer Pro to Business Unlimited. Require priority support.", "customer_email": "it.director@example.com", "customer_name": "Najd Hospital"},
  {"title": "5G speed under 10Mbps", "description": "In King Abdullah Financial District with full bars. Speedtest shows worse than 4G performance.", "customer_email": "finance.pro@example.com", "customer_name": "Khalid Investors", "location": {"latitude": 24.7584, "longitude": 46.6424}},
  {"title": "Duplicate bill for same period", "description": "Charged twice for January service. Reference # INV-78945 and INV-78946 show identical items.", "customer_email": "accounting@example.com", "customer_name": "Al-Marah Trading"},
  {"title": "WiFi calling drops after 2 minutes", "description": "Calls disconnect in basement office despite strong WiFi. Critical for business continuity.", "customer_email": "facility.manager@example.com", "customer_name": "Al-Nakheel Mall"},
  {"title": "Prepaid balance expired early", "description": "100 SAR balance disappeared after 15 days despite 30-day policy. Receipt # STC-2024-45612.", "customer_email": "student@example.com", "customer_name": "Mohammed Student"},
  {"title": "Fiber installation delay", "description": "Third missed appointment! Promised installation on 15/03 still not completed. Project deadline impacted.", "customer_email": "architect@example.com", "customer_name": "Design Studio", "location": {"latitude": 26.4207, "longitude": 50.0888}},
  {"title": "Corporate account hacking attempt", "description": "Multiple failed logins from foreign IP. Need immediate account freeze and investigation.", "customer_email": "cyber@example.com", "customer_name": "Bank Security Team"},
  {"title": "Gaming package throttling", "description": "After 50GB, speed drops to 1Mbps despite 'unlimited' claim. Ping becomes unplayable (>300ms).", "customer_email": "esports@example.com", "customer_name": "Saudi Gamers Club"},
  {"title": "STC Pay payment failure", "description": "Error 'Processor Declined' when paying 2,450 SAR bill. Card works elsewhere. Urgent to avoid suspension.", "customer_email": "urgent.payment@example.com", "customer_name": "Restaurant Chain"},
  {"title": "TV channel missing after renewal", "description": "BeIN Sports 1-12 disappeared after auto-renewal. Package shows active but channels not available.", "customer_email": "sports.fan@example.com", "customer_name": "Faisal Sports Bar"},
  {"title": "Business SMS rejection", "description": "Approved marketing messages now failing with 'Content Restricted'. Campaign is time-sensitive.", "customer_email": "marketing@example.com", "customer_name": "E-Commerce Startup"},
  {"title": "Frequent call drops in Dahran", "description": "Every call ends after 3-5 minutes in Al-Khobar area. Multiple devices affected.", "customer_email": "oil.engineer@example.com", "customer_name": "Aramco Contractor", "location": {"latitude": 26.2361, "longitude": 50.0393}},
  {"title": "VAT exemption not applied", "description": "Government account still charged 15% VAT despite valid exemption certificate on file.", "customer_email": "gov.account@example.com", "customer_name": "Ministry of Education"},
  {"title": "Router admin access blocked", "description": "Can't access 192.168.1.1 after firmware update. Need to configure port forwarding for security cameras.", "customer_email": "smart.home@example.com", "customer_name": "Villa Owner"},
  {"title": "Hajj package auto-renewal", "description": "Unwanted 500 SAR charge for Hajj package renewal. Was supposed to be one-time purchase.", "customer_email": "pilgrim@example.com", "customer_name": "Hajj Group 1445"},
  {"title": "Employee SIM abuse", "description": "Company number used for 8,000 SAR international calls after employee termination.", "customer_email": "hr.director@example.com", "customer_name": "Construction Company"
  }
]

headers = {
  'Content-Type': 'application/json'
}

for i in payload:
    response = requests.request("POST", url, headers=headers, data=json.dumps(i))
    print(response.text)