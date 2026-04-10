import asyncio
import structlog
from src.core.database import get_db_session_context
from src.core.models import Company, Product

# Setup logging
logger = structlog.get_logger()


async def seed_data():
    """Seed the database with sample company and product data."""
    logger.info("Starting database seeding...")

    async with get_db_session_context() as db:
        # 1. Create Sample Companies
        tech_corp = Company(
            name="TechNexus Solutions",
            industry="Software & AI",
            headquarters="San Francisco, CA",
        )

        green_energy = Company(
            name="EcoPulse Energy",
            industry="Renewable Energy",
            headquarters="Austin, TX",
        )

        db.add_all([tech_corp, green_energy])

        # Flush to get the IDs before creating products
        await db.flush()

        # 2. Create Sample Products
        products = [
            Product(
                company_id=tech_corp.id,
                name="NexusAI Platform",
                category="Enterprise Software",
                price=1200.00,
                description="An end-to-end machine learning orchestration platform.",
            ),
            Product(
                company_id=tech_corp.id,
                name="CodeStream IDE",
                category="Developer Tools",
                price=45.00,
                description="Cloud-native IDE for distributed teams.",
            ),
            Product(
                company_id=green_energy.id,
                name="SolarMax Panel v3",
                category="Hardware",
                price=850.50,
                description="High-efficiency monocrystalline solar panel.",
            ),
            Product(
                company_id=green_energy.id,
                name="GridGuard Battery",
                category="Storage",
                price=2400.00,
                description="Residential energy storage solution with 10kWh capacity.",
            ),
        ]

        db.add_all(products)

        try:
            await db.commit()
            logger.info(
                "Seeding completed successfully!",
                companies_added=2,
                products_added=len(products),
            )
        except Exception as e:
            await db.rollback()
            logger.error("Seeding failed", error=str(e))
            raise


if __name__ == "__main__":
    asyncio.run(seed_data())
