import asyncio
import socket
from bscpylgtv import WebOsClient
import os
import json
from config import config

from .tool_registry import tool, tool_registry

logger = config.get_logger("WebOS")

# Load credentials from JSON file
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'webos_creds.json'), 'r') as f:
    creds = json.load(f)

TV_IP = creds["ip_address"]
TV_MAC = creds["mac_address"]

class LGTVController:
    def __init__(self, tv_ip, mac_address=None):
        self.tv_ip = tv_ip
        self.mac_address = mac_address
        self.client = None
    
    async def connect(self):
        """Connect to the TV"""
        try:
            self.client = await WebOsClient.create(
                self.tv_ip, 
                ping_interval=None, 
                states=[]
            )
            await self.client.connect()
            logger.debug(f"✅ Connected to TV at {self.tv_ip}")
        except Exception as e:
            logger.debug(f"❌ Failed to connect to TV: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from the TV"""
        if self.client:
            await self.client.disconnect()
            logger.debug("🔌 Disconnected from TV")

    def wake_on_lan(self):
        """Turn on TV using Wake-on-LAN"""
        if not self.mac_address:
            logger.debug("❌ MAC address required for Wake-on-LAN")
            return False
            
        try:
            # Remove any separators from MAC address
            mac = self.mac_address.replace(':', '').replace('-', '').upper()
            
            # Create magic packet
            magic_packet = 'FF' * 6 + mac * 16
            magic_packet = bytes.fromhex(magic_packet)
            
            # Send magic packet
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            # Send to broadcast address
            # broadcast_ip = self.tv_ip.rsplit('.', 1)[0] + '.255'
            # sock.sendto(magic_packet, (broadcast_ip, 9))
            # Direct to TV IP
            sock.sendto(magic_packet, (self.tv_ip, 9))  
            sock.close()         

            logger.debug(f"📺 Wake-on-LAN packet sent to {self.mac_address}")
            return True
            
        except Exception as e:
            logger.debug(f"❌ Failed to send Wake-on-LAN: {e}")
            return False
    
    async def power_on(self):
        """Turn on TV using Wake-on-LAN, then establish connection"""
        logger.debug("🔌 Attempting to turn on TV...")
        
        # Send Wake-on-LAN packet
        if not self.wake_on_lan():
            return "Unable to turn on TV"
        
        # Wait for TV to boot up
        logger.debug("⏳ Waiting for TV to start...")
        await asyncio.sleep(10)  # Give TV time to boot
        
        # Try to establish connection
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                await self.connect()
                return "TV is now on"
            except Exception:
                logger.debug(f"🔄 Connection attempt {attempt + 1}/{max_attempts} failed, retrying...")
                await asyncio.sleep(3)

        return "TV may be on but connection failed"
    
    async def power_off(self):
        """Turn the TV off (standby)"""
        try:
            await self.connect()
            await self.client.power_off()
            return "TV is now off"
        except Exception as e:
            logger.debug(f"❌ Failed to turn off TV: {e}")
            return "Failed to turn off TV"
    
    async def volume_up(self):
        """Increase volume"""
        try:
            await self.connect()
            result = await self.client.volume_up()
            return f"Volume increased to {result}"
        except Exception as e:
            logger.debug(f"❌ Failed to increase volume: {e}")
            return "Failed to increase volume"
    
    async def volume_down(self):
        """Decrease volume"""
        try:
            await self.connect()
            result = await self.client.volume_down()
            return f"Volume decreased to {result}"
        except Exception as e:
            logger.debug(f"❌ Failed to decrease volume: {e}")
            return "Failed to decrease volume"
    
    async def set_volume(self, level):
        """Set volume to specific level (0-100)"""
        try:
            await self.connect()
            result = await self.client.set_volume(level)
            return f"Volume set to {level}"
        except Exception as e:
            logger.debug(f"❌ Failed to set volume: {e}")
            return "Failed to set volume"
    
    async def launch_netflix(self):
        """Launch Netflix app"""
        try:
            await self.connect()
            netflix_app_id = "netflix"
            await self.client.launch_app(netflix_app_id)
            logger.debug("🍿 Netflix launched")
            return "Netflix launched"
        except Exception as e:
            logger.debug(f"❌ Failed to launch Netflix: {e}")
            return "Failed to launch Netflix"

@tool(
    name="turn_on_tv",
    description="Turn on the TV",
    aliases=["tv_on", "watch_tv", "tv"]
)
def turn_on_tv():
    tv = LGTVController(TV_IP, TV_MAC)
    """Turn on the TV"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(tv.power_on())
    else:
        return loop.create_task(tv.power_on())
    
@tool(
    name="turn_off_tv",
    description="Turn off the TV",
    aliases=["tv_off", "no_tv"]
)
def turn_off_tv():
    tv = LGTVController(TV_IP, TV_MAC)
    """Turn off the TV"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(tv.power_off())
    else:
        return loop.create_task(tv.power_off())

@tool(
    name="set_tv_volume",
    description="Set TV Volume",
    aliases=["tv_volume", "volume_tv"]
)
def set_tv_volume(new_volume: str):
    tv = LGTVController(TV_IP, TV_MAC)
    """Set TV Volume"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(tv.set_volume(int(new_volume)))
    else:
        return loop.create_task(tv.set_volume(int(new_volume)))

@tool(
    name="launch_netflix",
    description="Launches Netflix",
    aliases=["netflix"]
)
def launch_netflix():
    tv = LGTVController(TV_IP, TV_MAC)
    """Launch Netflix"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(tv.launch_netflix())
    else:
        return loop.create_task(tv.launch_netflix())

# Testing
async def power_on_demo():
        tv = LGTVController(TV_IP, TV_MAC)
        
        try:
            # Turn on TV from standby
            success = await tv.power_on()
            
            if success:
                # Wait a moment, then launch Netflix
                await asyncio.sleep(2)
                await tv.launch_netflix()
                
                # Wait 10 seconds, then turn volume to 15
                await asyncio.sleep(10)
                await tv.set_volume(15)
                
                # Wait 5 seconds, then turn off
                await asyncio.sleep(5)
                await tv.power_off()
            
        except Exception as e:
            logger.debug(f"❌ Error: {e}")
        finally:
            await tv.disconnect()

if __name__ == "__main__":    
    logger.debug("🎯 LG webOS TV Controller")
    logger.debug("=" * 30)
    
    asyncio.run(power_on_demo())
