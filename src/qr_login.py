import os
import json
import asyncio
from telethon import TelegramClient, errors
from dotenv import load_dotenv
import qrcode

load_dotenv()

async def login_with_qr():
    # 1. Load your credentials
    api_id = int(os.getenv("TELEGRAM_ID", "") or "")
    api_hash = os.getenv("TELEGRAM_HASH", "")
    if not api_id or not api_hash:
        raise EnvironmentError("TELEGRAM_ID and TELEGRAM_HASH must be set in the .env file.")
    print(api_id, api_hash)
    
    # 2. Initialize a fresh client session template
    client = TelegramClient('scraper', api_id, api_hash)
    await client.connect()
    
    if await client.is_user_authorized():
        print("✓ Session is already authorized!")
        await client.disconnect()
        return

    # 3. Request a QR login session token from Telegram
    qr_login = await client.qr_login()
    print("\n" + "="*50)
    print(" SCAN THE QR CODE BELOW USING YOUR TELEGRAM MOBILE APP")
    print("="*50 + "\n")
    
    # Print the terminal-rendered visual QR code matrix
    qr = qrcode.QRCode()
    qr.add_data(qr_login.url)
    qr.print_ascii(tty=True)
    
    print("\n" + "="*50)
    print("Steps: Open Telegram -> Settings -> Devices -> Link Desktop Device")
    print("Waiting for your scan...")
    print("="*50 + "\n")


    try:
        # 4. Wait for the phone app to register the scan
        await qr_login.wait()
        print("✓ QR Scan Detected! Successfully authenticated!")
        
    except errors.SessionPasswordNeededError:
        # 5. CATCH 2FA: Prompt for the cloud password if enabled on your account
        print("\n🔒 Two-Step Verification is enabled on your account.")
        password = input("Please enter your Telegram Cloud Password: ")
        
        try:
            # Complete the login handshake passing the password parameters
            await client.sign_in(password=password)
            print("✓ Password accepted! Successfully authenticated via 2FA.")
        except Exception as password_error:
            print(f"❌ Failed to authenticate password: {password_error}")
            
    except Exception as e:
        print(f"❌ Authentication failed or expired: {e}")
        
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(login_with_qr())
